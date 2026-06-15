#pragma once

#include "infinicore/tensor.hpp"
#include <functional>
#include <string>
#include <memory>
#include <unordered_map>

namespace infinilm::models::debug_utils {

// TODO: move to InfiniCore as common utils in future work

/**
 * @brief Hook callback type for capturing intermediate values (DEBUG ONLY)
 *
 * Hook functions are called with:
 * - name: Identifier for the intermediate value (e.g., "layer0_q_after_proj")
 * - tensor: The intermediate tensor value
 * - layer_idx: Layer index (for layer-specific hooks, -1 if not applicable)
 *
 * NOTE: This is a debug utility. Do not use in production code.
 */
using HookCallback = std::function<void(const std::string &name, const infinicore::Tensor &tensor, int layer_idx)>;

/**
 * @brief Hook registry for managing hooks (DEBUG ONLY)
 *
 * NOTE: This is a debug utility for capturing intermediate tensor values
 * during model execution. Do not use in production code.
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

/**
 * @brief Macro to simplify hook registration (DEBUG ONLY)
 *
 * Usage: REGISTER_HOOK(registry, "hook_name", callback)
 */
#define REGISTER_HOOK(registry, name, callback) \
    (registry)->register_hook(name, callback)

/**
 * @brief Macro to simplify hook calls with automatic null and has_hooks checks (DEBUG ONLY)
 *
 * Usage: CALL_HOOK(registry, "hook_name", tensor)
 *        Note: layer_idx defaults to -1
 */
#define CALL_HOOK(registry, name, tensor) \
    do { \
        if ((registry) && (registry)->has_hooks()) { \
            (registry)->call_hook(name, tensor, -1); \
        } \
    } while (0)

/**
 * @brief Macro to simplify hook calls with explicit layer index (DEBUG ONLY)
 *
 * Usage: CALL_HOOK_LAYER(registry, "hook_name", tensor, layer_idx)
 */
#define CALL_HOOK_LAYER(registry, name, tensor, layer_idx) \
    do { \
        if ((registry) && (registry)->has_hooks()) { \
            (registry)->call_hook(name, tensor, layer_idx); \
        } \
    } while (0)

/**
 * @brief Macros to simplify hook_registry and hook_prefix management in model classes
 */

// Declare hook_registry and hook_prefix member variables
#define HOOK_REGISTRY_MEMBER() \
    std::shared_ptr<debug_utils::HookRegistry> hook_registry_; \
    std::string hook_prefix_;

// Set hook_registry and hook_prefix (no forwarding to submodules)
#define SET_HOOK_REGISTRY_SIMPLE() \
    void set_hook_registry(const std::shared_ptr<debug_utils::HookRegistry> &hook_registry, const std::string &hook_prefix = "") { \
        hook_registry_ = hook_registry; \
        hook_prefix_ = hook_prefix; \
    }

// Helper macro to build incremental hook prefix
#define BUILD_HOOK_PREFIX(prefix, name) \
    (prefix.empty() ? std::string(name) : prefix + "_" + std::string(name))

// Set hook_registry and hook_prefix and forward to one or more submodules
// Usage: SET_HOOK_REGISTRY(submodule1) or SET_HOOK_REGISTRY(submodule1, submodule2)
// The hook_prefix will be incremented for each submodule (e.g., "layer0" -> "layer0_attention")
// Note: Currently supports up to 2 submodules. For more, extend the pattern below.
#define SET_HOOK_REGISTRY(...) \
    SET_HOOK_REGISTRY_IMPL(__VA_ARGS__)

// Helper to handle variable number of arguments using a reliable pattern
#define SET_HOOK_REGISTRY_IMPL(...) \
    SET_HOOK_REGISTRY_GET_NTH(__VA_ARGS__, SET_HOOK_REGISTRY_2, SET_HOOK_REGISTRY_1, SET_HOOK_REGISTRY_0,)(__VA_ARGS__)

// Get the selector based on argument count
// Pattern: when we have N args, the (N+1)th parameter from the end is the selector
// For 0 args: _1=SET_HOOK_REGISTRY_2, _2=SET_HOOK_REGISTRY_1, _3=SET_HOOK_REGISTRY_0, N=(empty) → need to use _3
// For 1 arg: _1=arg, _2=SET_HOOK_REGISTRY_2, _3=SET_HOOK_REGISTRY_1, N=SET_HOOK_REGISTRY_0 → wrong, need _3
// For 2 args: _1=arg1, _2=arg2, _3=SET_HOOK_REGISTRY_2, N=SET_HOOK_REGISTRY_1 → wrong, need _3

// Use _3 as the selector (it's in the right position for all cases)
#define SET_HOOK_REGISTRY_GET_NTH(_1, _2, _3, N, ...) _3

// Implementation for 0 args (shouldn't be used, but handle gracefully)
#define SET_HOOK_REGISTRY_0() \
    void set_hook_registry(const std::shared_ptr<debug_utils::HookRegistry> &hook_registry, const std::string &hook_prefix = "") { \
        hook_registry_ = hook_registry; \
        hook_prefix_ = hook_prefix; \
    }

// Implementation for 1 arg
#define SET_HOOK_REGISTRY_1(submodule) \
    void set_hook_registry(const std::shared_ptr<debug_utils::HookRegistry> &hook_registry, const std::string &hook_prefix = "") { \
        hook_registry_ = hook_registry; \
        hook_prefix_ = hook_prefix; \
        if (submodule##_) { \
            std::string submodule_prefix = BUILD_HOOK_PREFIX(hook_prefix, #submodule); \
            submodule##_->set_hook_registry(hook_registry, submodule_prefix); \
        } \
    }

// Implementation for 2 args
#define SET_HOOK_REGISTRY_2(submodule1, submodule2) \
    void set_hook_registry(const std::shared_ptr<debug_utils::HookRegistry> &hook_registry, const std::string &hook_prefix = "") { \
        hook_registry_ = hook_registry; \
        hook_prefix_ = hook_prefix; \
        if (submodule1##_) { \
            std::string submodule1_prefix = BUILD_HOOK_PREFIX(hook_prefix, #submodule1); \
            submodule1##_->set_hook_registry(hook_registry, submodule1_prefix); \
        } \
        if (submodule2##_) { \
            std::string submodule2_prefix = BUILD_HOOK_PREFIX(hook_prefix, #submodule2); \
            submodule2##_->set_hook_registry(hook_registry, submodule2_prefix); \
        } \
    }

// Set hook_registry and hook_prefix for a vector of submodules
// For vectors, the prefix is incremented with an index (e.g., "layer0", "layer1", ...)
// If parent has a prefix, it becomes "parent_layer0", "parent_layer1", etc.
#define SET_HOOK_REGISTRY_VEC(vec_name) \
    void set_hook_registry(const std::shared_ptr<debug_utils::HookRegistry> &hook_registry, const std::string &hook_prefix = "") { \
        hook_registry_ = hook_registry; \
        hook_prefix_ = hook_prefix; \
        for (size_t i = 0; i < vec_name##_.size(); ++i) { \
            if (vec_name##_[i]) { \
                std::string layer_name = "layer" + std::to_string(i); \
                std::string item_prefix = BUILD_HOOK_PREFIX(hook_prefix, layer_name); \
                vec_name##_[i]->set_hook_registry(hook_registry, item_prefix); \
            } \
        } \
    }

} // namespace infinilm::models::debug_utils
