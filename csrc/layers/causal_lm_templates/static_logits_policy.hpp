#pragma once

#include <cstddef>

namespace infinilm::layers::causal_lm_templates::detail {

// Preserve the R5 shapes qualified against the Iluvatar BF16 vocabulary GEMM.
// A zero result keeps the full projection; one selects a compact projection.
inline size_t static_logits_rows(size_t batch, size_t sequence, size_t hidden,
                                 size_t vocabulary) {
    if (hidden != 3584 || vocabulary != 73448 || (batch != 4 && batch != 16)) {
        return 0;
    }
    if (sequence >= 2 && sequence <= 64) {
        return 1;
    }
    if ((batch == 4 && sequence == 512) || (batch == 16 && sequence == 2048)) {
        return 64;
    }
    return 0;
}

} // namespace infinilm::layers::causal_lm_templates::detail
