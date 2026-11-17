#pragma once

#include <cstddef>

namespace infinilm::global_state::hang_trace {

inline bool enabled() { return false; }

struct ScopedBracket {
    ScopedBracket(const char *, int = 0) {}
};

} // namespace infinilm::global_state::hang_trace
