#include "native_ops.hpp"

#include "execution/submit.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {

void apply_rotary_tail(runtime::Runtime &runtime, const runtime::Tensor &input,
                       const runtime::Tensor &positions, const runtime::Tensor &cos_sin, bool inverse) {
    using runtime::DataType;
    if (!input || (input->ndim() != 2 && input->ndim() != 3) || !input->is_contiguous()
        || (input->dtype() != DataType::kFloat32 && input->dtype() != DataType::kBFloat16)
        || !positions || positions->shape() != runtime::Shape{input->size(0)}
        || !positions->is_contiguous()
        || (positions->dtype() != DataType::kInt32 && positions->dtype() != DataType::kInt64)
        || !cos_sin || cos_sin->ndim() != 2 || !cos_sin->size(0) || cos_sin->size(1) != 64
        || cos_sin->dtype() != DataType::kFloat32 || !cos_sin->is_contiguous()) {
        throw std::invalid_argument("Invalid DeepSeek V4 tail rotary inputs");
    }
    const auto dim = static_cast<int64_t>(input->size(input->ndim() - 1));
    if (dim != 128 && dim != 512) {
        throw std::invalid_argument("DeepSeek V4 rotary input head must be 128 or 512 wide");
    }
    auto ids = positions;
    if (positions->dtype() == DataType::kInt32) {
        ids = runtime::Tensor::empty(runtime, positions->shape(), DataType::kInt64);
        using Cast = infini::ops::Cast;
        execution::submit<Cast>(runtime, execution::RecordingPolicy::DeviceOnly,
                                Cast::DefaultConfig(runtime.device().type()), positions, ids);
    }
    using Op = infini::ops::RotaryEmbedding;
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), ids, input,
                          std::optional<runtime::Tensor>{}, cos_sin, dim, false, dim - 64, inverse);
}

void store_mla_rows(runtime::Runtime &runtime, const runtime::Tensor &kv,
                    const runtime::Tensor &slots, const runtime::Tensor &pages,
                    size_t page_tokens) {
    if (page_tokens != 256 && page_tokens != 64 && page_tokens != 2) {
        throw std::invalid_argument("Unsupported DeepSeek V4 packed cache page size");
    }
    using Op = infini::ops::DeepseekV4StoreMla;
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), kv, slots,
                          static_cast<int64_t>(page_tokens), pages);
}

runtime::Tensor sparse_mla(runtime::Runtime &runtime, const runtime::Tensor &query,
                           const LayerPagedCache &cache, const runtime::Tensor &swa_indices,
                           const runtime::Tensor &swa_lengths, const runtime::Tensor &sink,
                           const runtime::Tensor &compressed_indices,
                           const runtime::Tensor &compressed_lengths, double scale) {
    const auto ratio = cache.compression_ratio;
    if (!query || (ratio != 0 && ratio != 4 && ratio != 128)
        || static_cast<bool>(ratio) != static_cast<bool>(cache.compressed)
        || static_cast<bool>(ratio) != static_cast<bool>(compressed_indices)
        || static_cast<bool>(ratio) != static_cast<bool>(compressed_lengths)) {
        throw std::invalid_argument("DeepSeek V4 compressed cache and sparse metadata disagree");
    }
    using Op = infini::ops::DeepseekV4SparseMla;
    using Opt = std::optional<runtime::Tensor>;
    auto output = runtime::Tensor::empty(runtime, query->shape(), query->dtype());
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), query, cache.swa,
                          swa_indices, swa_lengths, sink, ratio ? Opt{cache.compressed} : Opt{},
                          ratio ? Opt{compressed_indices} : Opt{},
                          ratio ? Opt{compressed_lengths} : Opt{},
                          static_cast<int64_t>(ratio ? kV4PageSize / ratio : 0), scale, output);
    return output;
}

runtime::Tensor run_routed_experts(runtime::Runtime &runtime, const runtime::Tensor &input,
                                   const RoutedExperts &routing, const W8A8ExpertWeights &weights,
                                   double swiglu_limit, std::optional<double> routed_scale) {
    if (!input) throw std::invalid_argument("DeepSeek V4 routed experts require input");
    using Op = infini::ops::DeepseekV4W8a8Experts;
    auto output = runtime::Tensor::empty(runtime, input->shape(), input->dtype());
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), input, routing.weights,
                          routing.indices, weights.gate_up(), weights.gate_up_scale(),
                          weights.down(), weights.down_scale(), swiglu_limit, routed_scale, output);
    return output;
}

} // namespace infinilm::models::deepseek_v4
