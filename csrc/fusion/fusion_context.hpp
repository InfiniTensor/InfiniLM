/**
 * @file fusion_context.hpp
 * @brief Thread-local fusion context for dynamic Python → C++ fusion decisions
 *
 * This class provides a mechanism for Python to communicate per-forward
 * fusion decisions to C++ execution layer.
 *
 * Usage:
 *   Python: FusionContext.set("add_rms_norm", True)
 *   C++:    if (FusionContext::get("add_rms_norm")) { ... }
 */

#pragma once

#include <string>
#include <unordered_map>

namespace infinilm::fusion {

/**
 * @brief Thread-local context for fusion decisions.
 *
 * Python sets fusion decisions before calling forward(),
 * C++ layers read these decisions during execution.
 */
class FusionContext {
public:
    /**
     * @brief Set fusion decision for an operation.
     * @param op_name Operation name (e.g., "add_rms_norm", "swiglu")
     * @param should_fuse Whether to use fused kernel
     */
    static void set(const std::string &op_name, bool should_fuse);

    /**
     * @brief Get fusion decision for an operation.
     * @param op_name Operation name
     * @param default_value Default value if not set (default: true)
     * @return Whether to use fused kernel
     */
    static bool get(const std::string &op_name, bool default_value = true);

    /**
     * @brief Check if fusion decision is explicitly set for an operation.
     * @param op_name Operation name
     * @return true if decision is set, false otherwise
     */
    static bool has(const std::string &op_name);

    /**
     * @brief Clear all fusion decisions.
     * Should be called after forward() completes.
     */
    static void clear();

    /**
     * @brief Get number of decisions currently set.
     */
    static size_t size();

private:
    // Thread-local storage for fusion decisions
    static thread_local std::unordered_map<std::string, bool> decisions_;
};

} // namespace infinilm::fusion
