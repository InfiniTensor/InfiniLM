#pragma once

#include "submit.hpp"

namespace infinilm::execution::paged_cache_checks {

// LM owns association and empty-request policy. Empty never selects a provider.
inline std::optional<infini::ops::Config> validate(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &kc,
    const infinilm::runtime::Tensor &vc, const infinilm::runtime::Tensor &k,
    const infinilm::runtime::Tensor &v, const infinilm::runtime::Tensor &slots) {
    runtime.assertOwnerThread();
    for (const auto &tensor : {kc, vc, k, v, slots}) {
        detail::check_owner(runtime, tensor);
    }
    if (!slots->size(0)) {
        return std::nullopt;
    }
    return infini::ops::ReshapeAndCacheFlash::DefaultConfig(runtime.device().type());
}

} // namespace infinilm::execution::paged_cache_checks
