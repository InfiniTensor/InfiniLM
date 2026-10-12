#include "deepep_dispatcher.hpp"

#include "runtime/runtime.hpp"
#include <stdexcept>

namespace infinilm::layers::moe {

DeepEPDispatcher::DeepEPDispatcher(EPConfig config)
    : config_(config) {}

DispatchOutput DeepEPDispatcher::dispatch(infinilm::runtime::Runtime &runtime,
                                          const infinilm::runtime::Tensor &hidden_states,
                                          const TopKOutput &topk_output,
                                          MoeWorkspace &workspace) const {
    runtime.assertOwnerThread();
    (void)workspace;
    dispatch_a(hidden_states, topk_output);
    return dispatch_b();
}

infinilm::runtime::Tensor DeepEPDispatcher::combine(infinilm::runtime::Runtime &runtime,
                                             const CombineInput &combine_input,
                                             MoeWorkspace &workspace) const {
    runtime.assertOwnerThread();
    (void)workspace;
    combine_a(combine_input);
    return combine_b();
}

void DeepEPDispatcher::dispatch_a(const infinilm::runtime::Tensor &hidden_states,
                                  const TopKOutput &topk_output) const {
    (void)config_;
    (void)hidden_states;
    (void)topk_output;
    throw std::runtime_error("DeepEPDispatcher::dispatch_a is reserved but not implemented yet");
}

DispatchOutput DeepEPDispatcher::dispatch_b() const {
    throw std::runtime_error("DeepEPDispatcher::dispatch_b is reserved but not implemented yet");
}

void DeepEPDispatcher::combine_a(const CombineInput &combine_input) const {
    (void)config_;
    (void)combine_input;
    throw std::runtime_error("DeepEPDispatcher::combine_a is reserved but not implemented yet");
}

infinilm::runtime::Tensor DeepEPDispatcher::combine_b() const {
    throw std::runtime_error("DeepEPDispatcher::combine_b is reserved but not implemented yet");
}

} // namespace infinilm::layers::moe
