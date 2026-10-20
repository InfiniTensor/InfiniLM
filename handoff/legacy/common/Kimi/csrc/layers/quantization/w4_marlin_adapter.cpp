#include "w4_marlin_adapter.hpp"
#include "../../execution/submit.hpp"
#include "runtime/runtime.hpp"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <exception>
#include <numeric>
#include <stdexcept>

namespace infinilm::quantization {
using infinilm::runtime::DataType;
using infinilm::runtime::Runtime;
using infinilm::runtime::Tensor;
namespace { void require(bool ok, const char* msg) { if (!ok) throw std::invalid_argument(msg); } }
namespace w4_detail {
void drain_or_terminate(Runtime& runtime) noexcept {
    try { runtime.syncStream(); } catch (...) { std::terminate(); }
}
Tensor contiguous(Runtime& runtime, const Tensor& source) {
    runtime.assertOwnerThread();
    require(source && source->device() == runtime.device(), "W4 contiguous Runtime/device mismatch");
    if (source->is_contiguous()) return source;
    // Own destination outside try. Do not invoke Tensor::contiguous's internal
    // destination, which the caller cannot retain if Rearrange throws.
    Tensor destination;
    try {
        destination = Tensor::empty(runtime, source->shape(), source->dtype());
        infinilm::execution::submit<infini::ops::Copy>(
            runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
            source, false, destination);
        return destination;
    } catch (...) { drain_or_terminate(runtime); throw; }
}
}
namespace {
std::vector<int32_t> read_i32(Runtime& runtime, const Tensor& source) {
    require(source && source->dtype() == DataType::kInt32 && source->device() == runtime.device(),
            "W4 expected int32 checkpoint tensor on Runtime device");
    Tensor contiguous;
    std::vector<int32_t> result;
    try {
        contiguous = w4_detail::contiguous(runtime, source);
        result.resize(contiguous->numel());
        if (!result.empty()) runtime.memcpyD2H(result.data(), contiguous->data(), result.size()*sizeof(int32_t));
        runtime.syncStream();
        return result;
    } catch (...) { w4_detail::drain_or_terminate(runtime); throw; }
}
Tensor upload_i32(Runtime& runtime, const std::vector<int32_t>& values) {
    Tensor result;
    try {
        result = Tensor::empty(runtime, {values.size()}, DataType::kInt32);
        if (!values.empty()) runtime.memcpyH2D(result->data(), values.data(), values.size()*sizeof(int32_t), false);
        runtime.syncStream(); // caller-owned host vector cannot retire before completion
        return result;
    } catch (...) { w4_detail::drain_or_terminate(runtime); throw; }
}
}

w4_detail::Metadata w4_detail::validate_metadata(Runtime& runtime, const ParamsMap& raw,
    const nlohmann::json& config, int split_dim, bool retained_global_scale_table) {
    runtime.assertOwnerThread();
    require(split_dim >= -1 && split_dim <= 1, "W4 binding split dimension is invalid");
    require(runtime.graphCaptureState() == infinilm::execution::graph::GraphManager::CaptureState::kInactive,
            "W4 checkpoint preparation requires inactive graph state");
    require(runtime.device().type() == infinilm::runtime::Device::Type::kNvidia, "W4 Marlin requires NVIDIA; no fallback");
    require(config.is_object(), "W4 config must be an object");
    const auto method = config.at("quant_method").get<std::string>();
    const int scheme = method == "awq" ? 0 : method == "gptq" ? 1 : -1;
    require(scheme != -1, "W4 adapter only accepts AWQ/GPTQ");
    require(config.value("bits", config.value("w_bit", 4)) == 4, "W4 adapter requires exactly four bits");
    if (scheme == 0) require(config.value("version", std::string("gemm")) == "gemm" ||
        config.value("version", std::string("gemm")) == "GEMM", "AWQ GEMV packing is unsupported");
    const int group = config.value("group_size", config.value("q_group_size", 128));
    auto qw = raw.at("qweight"), scales = raw.at("scales"), zeros = raw.at("qzeros");
    require(qw && qw->ndim() == 2 && qw->dtype() == DataType::kInt32, "W4 qweight checkpoint shape/dtype invalid");
    require(qw->size(scheme == 0 ? 1 : 0) <= SIZE_MAX/8, "W4 packed dimension overflow");
    const size_t k = qw->size(0)*(scheme == 0 ? 1 : 8), n = qw->size(1)*(scheme == 0 ? 8 : 1);
    require(k && n && k%128 == 0 && n%64 == 0, "W4 Marlin requires local K%128=0/N%64=0");
    require(group == -1 || group == 32 || group == 64 || group == 128, "W4 unsupported group size");
    require(group == -1 || k%group == 0, "W4 shard must end on group boundary");
    require(scales && scales->ndim() == 2 && scales->size(1) == n && scales->size(0) > 0 &&
        (scales->dtype() == DataType::kFloat16 || scales->dtype() == DataType::kBFloat16), "W4 scales checkpoint shape/dtype invalid");
    const size_t groups = scales->size(0);
    const bool desc = scheme == 1 && config.value("desc_act", false);
    if (scheme == 1) {
        require(config.value("sym", true), "asymmetric GPTQ is unsupported by U4B8");
        require(config.contains("checkpoint_format") && config.at("checkpoint_format") == "gptq",
                "explicit historical GPTQ z+1 format is required");
        const auto ids = raw.at("g_idx");
        require(ids && ids->dtype() == DataType::kInt32 && ids->shape() == infinilm::runtime::Shape{k} &&
                ids->device() == runtime.device(), "GPTQ g_idx must be a local int32 K vector on Runtime device");
    }
    const bool full = !desc || split_dim != 1;
    require(full || retained_global_scale_table, "GPTQ desc_act row TP requires unsliced global scales");
    require(full || (group > 0 && size_t(group) < k), "partial-K GPTQ requires group_size < local K");
    require(!full || groups == (group == -1 ? 1 : k/group), "W4 scales group count mismatch");
    for (const auto& t : {qw,scales,zeros}) require(t && t->device() == runtime.device(), "W4 checkpoint device differs from Runtime");
    require(zeros->dtype() == DataType::kInt32 && zeros->shape() == infinilm::runtime::Shape{groups,n/8}, "W4 qzeros layout mismatch");
    require(n <= SIZE_MAX/2, "W4 packed output dimension overflow");
    require(k <= SIZE_MAX/n/4 && groups <= SIZE_MAX/n/4, "W4 allocation byte size overflow");
    if (raw.count("bias")) {
        const auto b = raw.at("bias");
        require(b && b->shape() == infinilm::runtime::Shape{n} &&
                b->dtype() == scales->dtype() && b->device() == runtime.device(),
                "W4 bias differs from local output contract");
    }
    return {k,n,groups,scheme,group,desc,full};
}

PreparedW4 prepare_w4_marlin(Runtime& runtime, const ParamsMap& raw, const nlohmann::json& config,
                            int split_dim, bool retained_global_scale_table,
                            std::optional<int64_t> expected_non_desc_group_offset) {
#ifdef INFINILM_W4_MARLIN_OPS_UNAVAILABLE
    runtime.assertOwnerThread();
    throw std::runtime_error("W4 Marlin public APIs are unavailable in this SDK; AWQ/GPTQ execution is unsupported");
#else

    const auto metadata = w4_detail::validate_metadata(runtime, raw, config, split_dim, retained_global_scale_table);
    const auto k = metadata.k, n = metadata.n, groups = metadata.groups;
    const auto scheme = metadata.scheme, group = metadata.group;
    const auto desc = metadata.desc, full = metadata.full;
    require(!expected_non_desc_group_offset ||
            (scheme == 1 && !desc && *expected_non_desc_group_offset >= 0),
            "GPTQ group offset constraint requires non-desc GPTQ and a nonnegative offset");
    auto qw = raw.at("qweight"), scales = raw.at("scales"), zeros = raw.at("qzeros");

    // All device destinations / original and contiguous sources survive catch.
    PreparedW4 result{};
    Tensor empty, perm, gidx, packed, reordered_scales, reordered_zeros, bias;
    std::vector<int32_t> ids, order, sorted;
    try {
        result.params = raw;
        result.k = k; result.n = n; result.scheme = scheme; result.group = group; result.full = full;
        empty = upload_i32(runtime, {}); perm = empty; gidx = empty;
        if (scheme == 1) {
            for (int32_t word : read_i32(runtime, zeros))
                require(uint32_t(word) == 0x77777777u, "GPTQ stored zero nibble must be seven (effective zero eight)");
            require(raw.at("g_idx") && raw.at("g_idx")->shape() == infinilm::runtime::Shape{k}, "GPTQ g_idx must be a local K vector");
            ids = read_i32(runtime, raw.at("g_idx"));
            require(ids.size() == k, "GPTQ g_idx length mismatch");
            if (desc) {
                for (auto id : ids) require(id >= 0 && size_t(id) < groups, "GPTQ g_idx outside retained scale table");
                require(group > 0 && (!full || groups > 1), "GPTQ desc_act requires complete groups");
                require(k <= size_t(INT32_MAX), "GPTQ permutation index overflow");
                order.resize(k); std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int32_t a,int32_t b){ return ids[a] < ids[b]; });
                sorted.resize(k);
                for (size_t i=0; i<k; ++i) sorted[i] = ids[order[i]];
                if (full) for (size_t i=0; i<k; ++i)
                    require(sorted[i] == int32_t(i/group), "full-K GPTQ group cardinality mismatch");
                perm = upload_i32(runtime, order); gidx = upload_i32(runtime, sorted);
            } else {
                const int64_t first = split_dim == 1 && group != -1 ? ids.front() : 0;
                require(first >= 0, "GPTQ group offset must be nonnegative");
                require(!expected_non_desc_group_offset || first == *expected_non_desc_group_offset,
                        "GPTQ group offset differs from constructor TP geometry");
                for (size_t i=0; i<k; ++i)
                    require(int64_t(ids[i]) == first + int64_t(group == -1 ? 0 : i/group), "GPTQ non-desc g_idx must be canonical and group-aligned");
            }
            zeros = empty;
        }
        qw = w4_detail::contiguous(runtime, qw);
        scales = w4_detail::contiguous(runtime, scales);
        zeros = w4_detail::contiguous(runtime, zeros);
        packed = Tensor::empty(runtime, {k/16,n*2}, DataType::kInt32);
        reordered_scales = Tensor::empty(runtime, {groups,n}, scales->dtype());
        reordered_zeros = scheme == 0 ? Tensor::empty(runtime, {groups,n/8}, DataType::kInt32) : empty;
        infinilm::execution::submit<infini::ops::W4MarlinRepack>(
            runtime, infinilm::execution::RecordingPolicy::HostOnly,
            infini::ops::Config::Automatic(), qw, perm,
            static_cast<int64_t>(scheme), static_cast<int64_t>(k), static_cast<int64_t>(n), packed);
        infinilm::execution::submit<infini::ops::W4MarlinParameters>(
            runtime, infinilm::execution::RecordingPolicy::HostOnly,
            infini::ops::Config::Automatic(), scales, zeros,
            static_cast<int64_t>(scheme), static_cast<int64_t>(k),
            static_cast<int64_t>(n), static_cast<int64_t>(group),
            reordered_scales, reordered_zeros);
        if (raw.count("bias")) bias = w4_detail::contiguous(runtime, raw.at("bias"));
        result.params["qweight"] = packed; result.params["scales"] = reordered_scales;
        result.params["qzeros"] = reordered_zeros; result.params["g_idx"] = gidx;
        result.params["perm"] = perm;
        result.params["empty_bias"] = Tensor::empty(runtime, {0}, scales->dtype());
        if (bias) result.params["bias"] = bias;
        runtime.syncStream(); // includes bias and all private map construction
        return result;
    } catch (...) { w4_detail::drain_or_terminate(runtime); throw; }

#endif
}
} // namespace infinilm::quantization
