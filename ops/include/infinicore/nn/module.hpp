#pragma once

#include "../tensor.hpp"
#include "parameter.hpp"

#include <spdlog/spdlog.h>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace infinicore::nn {
class Module {
public:
    Module() = default;

    const std::unordered_map<std::string, Parameter> &state_dict() const;

    void load_state_dict(const std::unordered_map<std::string, Tensor> &_state_dict);

    void load_parameter(const std::string &name, const Tensor &param);

    void load_parameter_(const std::string &name, const Tensor &param);

    void load_parameter_from_blob(const std::string &name, const void *data);

protected:
    Tensor register_parameter(const std::string &name, Parameter param);

    Tensor register_buffer(const std::string &name, Parameter buffer);

    // Add an existing submodule to this module's hierarchy
    // Template parameter M must be a type derived from Module
    // Returns the submodule for convenience (allows method chaining)
    template <typename M>
    std::shared_ptr<M> add_module(const std::string &name, std::shared_ptr<M> submodule) {
        // Ensure M is derived from Module (compile-time check)
        static_assert(std::is_base_of<Module, M>::value,
                      "Template parameter M must be derived from infinicore::nn::Module");

        // Store in the submodules map (std::shared_ptr<M> automatically converts to std::shared_ptr<Module>)
        submodules_[name] = submodule;

        return submodule;
    }

    // Create and register a new submodule by constructing it with the given arguments
    // Template parameter M must be a type derived from Module
    // Args are forwarded to M's constructor
    template <typename M, typename... Args>
    std::shared_ptr<M> register_module(const std::string &name, Args &&...args) {
        // Ensure M is derived from Module (compile-time check)
        static_assert(std::is_base_of<Module, M>::value,
                      "Template parameter M must be derived from infinicore::nn::Module");

        // Construct the submodule
        auto submodule = std::make_shared<M>(std::forward<Args>(args)...);

        return add_module(name, submodule);
    }

    // Create and register multiple submodules of the same type
    // Each submodule is named as "name.0", "name.1", etc.
    // Template parameter M must be a type derived from Module
    template <typename M, typename... Args>
    std::vector<std::shared_ptr<M>> register_modules(size_t count, const std::string &name, Args &&...args) {
        static_assert(std::is_base_of<Module, M>::value,
                      "Template parameter M must be derived from infinicore::nn::Module");

        std::vector<std::shared_ptr<M>> modules;
        modules.reserve(count);
        for (size_t i = 0; i < count; i++) {
            modules.push_back(register_module<M>(name + "." + std::to_string(i), std::forward<Args>(args)...));
        }
        return modules;
    }

protected:
    Device device_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;
    std::unordered_map<std::string, Parameter> buffers_;
    std::unordered_map<std::string, Parameter> parameters_;

private:
    void load_state_dict_recursively(const std::unordered_map<std::string, Tensor> &_state_dict, const std::string &prefix = "");
    void collect_all_parameters(std::unordered_map<std::string, Parameter> &all_params, const std::string &prefix = "") const;
};

// ============================================================================
// PyTorch-like Macros for Convenient Module Registration
// ============================================================================

/**
 * @brief Register submodules with automatic name inference from variable name
 *
 * Usage:
 * @code
 *   class MyModel : public Module {
 *   protected:
 *       INFINICORE_NN_MODULE(Linear, layer1);
 *       INFINICORE_NN_MODULE(Linear, layer2);
 *       INFINICORE_NN_MODULE_VEC(Linear, layers);
 *       INFINICORE_NN_PARAMETER(scaling_factor);
 *
 *   public:
 *       MyModel() {
 *           INFINICORE_NN_MODULE_INIT(layer1, 128, 64);
 *           INFINICORE_NN_MODULE_INIT(layer2, 64, 32);
 *           INFINICORE_NN_MODULE_VEC_INIT(layers, 3, Linear, 32, 16);
 *           INFINICORE_NN_PARAMETER_INIT(scaling_factor, ({1}, DataType::F32, Device()));
 *       }
 *   };
 * @endcode
 */

// Declare a single module member variable
#define INFINICORE_NN_MODULE(ModuleType, name) \
    std::shared_ptr<ModuleType> name##_

// Declare a vector of modules member variable
#define INFINICORE_NN_MODULE_VEC(ModuleType, name) \
    std::vector<std::shared_ptr<ModuleType>> name##_

// Initialize a module in constructor
#define INFINICORE_NN_MODULE_INIT(name, ...) \
    name##_ = this->register_module<std::remove_reference<decltype(*name##_)>::type>(#name, ##__VA_ARGS__)

// Initialize a vector of modules in constructor
// Usage: INFINICORE_NN_MODULE_VEC_INIT(layers, count, ModuleType, ctor_args...)
// Example: INFINICORE_NN_MODULE_VEC_INIT(layers, 3, Linear, 128, 64)
#define INFINICORE_NN_MODULE_VEC_INIT(name, count, ModuleType, ...) \
    name##_ = this->register_modules<ModuleType>(count, #name, ##__VA_ARGS__)

// Declare a parameter member variable
#define INFINICORE_NN_PARAMETER(name) \
    infinicore::nn::Parameter name##_

// Initialize a parameter in constructor
// Usage: INFINICORE_NN_PARAMETER_INIT(name, (shape, dtype, device))
// Example: INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, DataType::F32, device))
#define INFINICORE_NN_PARAMETER_INIT(name, args) \
    name##_ = infinicore::nn::Parameter args;    \
    this->register_parameter(#name, name##_)

// Declare a buffer member variable
#define INFINICORE_NN_BUFFER(name) \
    infinicore::nn::Parameter name##_

// Initialize a buffer in constructor
// Usage: INFINICORE_NN_BUFFER_INIT(name, (shape, dtype, device))
// Example: INFINICORE_NN_BUFFER_INIT(cache, ({max_seq_len, head_dim}, DataType::F32, device))
#define INFINICORE_NN_BUFFER_INIT(name, args) \
    name##_ = infinicore::nn::Parameter args; \
    this->register_buffer(#name, name##_)

} // namespace infinicore::nn
