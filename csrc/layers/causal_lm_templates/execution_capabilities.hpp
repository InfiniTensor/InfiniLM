#pragma once

#include <type_traits>

namespace infinilm::layers::causal_lm_templates {

// A module opts in only when splitting a causal request preserves its state and
// token semantics. Unknown, recurrent, and routed modules default to false.
template <typename Module, typename = void>
struct ChunkedPrefillCapability : std::false_type {};

template <typename Module>
struct ChunkedPrefillCapability<Module, std::void_t<decltype(Module::supports_chunked_prefill)>>
    : std::bool_constant<Module::supports_chunked_prefill> {};

} // namespace infinilm::layers::causal_lm_templates
