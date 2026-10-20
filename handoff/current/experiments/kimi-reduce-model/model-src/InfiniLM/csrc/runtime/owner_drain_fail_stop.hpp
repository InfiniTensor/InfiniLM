#pragma once
#include <exception>
namespace infinilm::runtime::detail {
// Used by the actual Runtime status sites and independent child-process fixture.
// No state/queue/registry. Failure is fatal only in the opt-in NSA owner domain.
inline void require_proven_owner_drain(bool strict_owner, bool success) noexcept {
    if (strict_owner && !success) {
        std::terminate();
    }
}
} // namespace infinilm::runtime::detail
