#pragma once
#include <exception>
#include <stdexcept>
#include <utility>

namespace infinilm::engine::nsa {
// Only call FROM a catch, with the exact caught exception_ptr. The caller keeps
// retained RUN locals outside that inner try/catch, so cleanup precedes release.
// No destructor captures current_exception(), no state, queue or registry.
template<class Abort, class Revoke, class Drain>
[[noreturn]] void retire_caught_run_failure(std::exception_ptr error,
                                          Abort &&abort, Revoke &&revoke, Drain &&drain) {
    if (!error) throw std::logic_error("NSA retirement requires a caught exception");
    std::forward<Abort>(abort)(error); // actual first error is published BEFORE peer wake/drain
    std::forward<Revoke>(revoke)();   // CPU validity only; strong backing remains held
    if (!std::forward<Drain>(drain)()) std::terminate(); // unfinished resources cannot unwind
    std::rethrow_exception(error);   // preserve type, payload and exception identity
}
} // namespace infinilm::engine::nsa
