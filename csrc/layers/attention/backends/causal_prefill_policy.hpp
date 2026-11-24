#ifndef INFINILM_CAUSAL_PREFILL_POLICY_HPP_
#define INFINILM_CAUSAL_PREFILL_POLICY_HPP_

#include <cstddef>
#include <string_view>

namespace infinilm::layers::attention::backends::detail {

inline bool UseFusedCausalPrefill(const char *flag, bool iluvatar, bool bf16,
                                  bool contiguous, std::size_t rank,
                                  std::size_t heads, std::size_t query,
                                  std::size_t key) {
    return flag != nullptr && std::string_view(flag) == "1" && iluvatar && bf16
        && contiguous && rank == 3 && heads > 0 && query > 1 && query <= key
        && key <= 4096 && heads <= (std::size_t{65535} * 65535) / query;
}

} // namespace infinilm::layers::attention::backends::detail

#endif
