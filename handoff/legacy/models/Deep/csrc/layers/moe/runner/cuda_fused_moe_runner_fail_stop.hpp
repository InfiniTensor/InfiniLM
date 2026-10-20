#pragma once
#include <cstdio>
#include <exception>

namespace infinilm::layers::moe::detail {
// No unwind, cancel, or Runtime reuse while completion is unknown. Same
// terminate policy as prepare_state_guard; private to this producer scope.
[[noreturn]] inline void fail_stop_unknown_runner_drain() noexcept {
    std::fputs("MoE runner fail-stop: stream completion unknown\n",stderr);
    std::fflush(stderr);
    std::terminate();
}
}
