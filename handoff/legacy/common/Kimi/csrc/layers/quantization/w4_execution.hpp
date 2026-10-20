#pragma once
#include "base_quantization.hpp"

namespace infinilm::quantization {
// One immutable execution binding; raw checkpoint registration stays in layer.
class W4Execution final {
public:
    static std::shared_ptr<const W4Execution> prepare(infinilm::runtime::Runtime&,
        const ParamsMap&, const nlohmann::json&, int split_dim,
        bool retained_global_scale_table = false,
        std::optional<int64_t> expected_non_desc_group_offset = std::nullopt);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime&, const infinilm::runtime::Tensor&,
        bool has_bias, float alpha = 1.f) const;
    infinilm::runtime::Tensor forward_allreduce(infinilm::runtime::Runtime&, const infinilm::runtime::Tensor&,
        bool has_bias, infinicclComm_t, float alpha = 1.f) const;
private:
    W4Execution(infinilm::runtime::Runtime&, const ParamsMap&, const ParamsMap&,
                size_t, size_t, int, int, bool);
    std::weak_ptr<infinilm::runtime::Runtime> owner_;
    ParamsMap raw_leases_;
    infinilm::runtime::Tensor weight_, scales_, zeros_, gidx_, perm_, bias_, empty_bias_;
    size_t k_, n_;
    int scheme_, group_;
    bool full_;
};
inline bool is_w4(QuantScheme scheme) {
    return scheme == QuantScheme::AWQ_W4A16 || scheme == QuantScheme::GPTQ_W4A16;
}
} // namespace infinilm::quantization
