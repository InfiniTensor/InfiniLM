#pragma once
#include "base_quantization.hpp"

namespace infinilm::quantization {
// Private loading result. No raw registration is replaced by preparation.
struct PreparedW4 {
    ParamsMap params;
    size_t k, n;
    int scheme, group;
    bool full;
};
PreparedW4 prepare_w4_marlin(infinilm::runtime::Runtime &, const ParamsMap &,
                             const nlohmann::json &, int split_dim,
                             bool retained_global_scale_table = false,
                             std::optional<int64_t> expected_non_desc_group_offset = std::nullopt);
namespace w4_detail {
struct Metadata {
    size_t k, n, groups;
    int scheme, group;
    bool desc, full;
};
// Pure loading-domain validation; neither device work nor a validation ticket.
Metadata validate_metadata(infinilm::runtime::Runtime &, const ParamsMap &,
                           const nlohmann::json &, int split_dim,
                           bool retained_global_scale_table);
void drain_or_terminate(infinilm::runtime::Runtime &) noexcept;
infinilm::runtime::Tensor contiguous(infinilm::runtime::Runtime &, const infinilm::runtime::Tensor &);
} // namespace w4_detail
} // namespace infinilm::quantization
