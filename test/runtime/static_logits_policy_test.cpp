#include "csrc/layers/causal_lm_templates/static_logits_policy.hpp"

#include <array>
#include <limits>

int main() {
    using infinilm::layers::causal_lm_templates::detail::static_logits_rows;
    for (size_t batch : {4, 16}) {
        for (size_t sequence : {2, 17, 63, 64}) {
            if (static_logits_rows(batch, sequence, 3584, 73448) != 1) {
                return 1;
            }
        }
    }
    if (static_logits_rows(4, 512, 3584, 73448) != 64
        || static_logits_rows(16, 2048, 3584, 73448) != 64) {
        return 2;
    }
    const std::array<std::array<size_t, 4>, 12> fallback{{{0, 512, 3584, 73448}, {1, 512, 3584, 73448}, {8, 512, 3584, 73448}, {4, 0, 3584, 73448}, {4, 1, 3584, 73448}, {4, 65, 3584, 73448}, {4, 511, 3584, 73448}, {4, 513, 3584, 73448}, {4, 2048, 3584, 73448}, {4, 512, 4096, 73448}, {4, 512, 3584, 32000}, {16, std::numeric_limits<size_t>::max(), 3584, 73448}}};
    for (const auto &shape : fallback) {
        if (static_logits_rows(shape[0], shape[1], shape[2], shape[3]) != 0) {
            return 3;
        }
    }
    return 0;
}
