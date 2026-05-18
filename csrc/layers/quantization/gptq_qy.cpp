#include "gptq_qy.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/linear_w4a16_gptq_qy.hpp"
#include <optional>

namespace infinilm::quantization {

std::vector<ParamDescriptor> GPTQ_QY::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    // GPTQ_QY weight layout is transposed: qweight [in_features/2, out_features]
    // ColumnParallel (split_dim=0, split output) → tp_dim=1
    // RowParallel    (split_dim=1, split input)  → tp_dim=0
    int gptq_tp_dim = (split_dim >= 0) ? (1 - split_dim) : -1;
    int group_size = get_group_size();

    std::vector<ParamDescriptor> descs;
    descs.push_back({"qweight", {in_features / 2, out_features}, infinicore::DataType::U8, gptq_tp_dim, tp_rank, tp_size});
    descs.push_back({"qzeros", {in_features / group_size, out_features}, dtype, gptq_tp_dim, tp_rank, tp_size});
    descs.push_back({"scales", {in_features / group_size, out_features}, dtype, gptq_tp_dim, tp_rank, tp_size});
    descs.push_back({"g_idx", {in_features}, infinicore::DataType::I32, -1, 0, 1});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor GPTQ_QY::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float /*alpha*/) const {
    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto qweight = params.at("qweight");
    auto qzeros = params.at("qzeros");
    auto scales = params.at("scales");

    auto output = infinicore::op::linear_w4a16_gptq_qy(input_contiguous->contiguous(), qweight, qzeros, scales, 0, 4);

    if (has_bias) {
        auto bias = params.at("bias");
        infinicore::op::add_(output, output, bias->as_strided(output->shape(), {0, 0, 1}));
    }
    return output;
}

std::vector<SplitParam> GPTQ_QY::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int /*narrow_dim*/,
    int tp_rank, int tp_size, int tp_num_heads) const {

    // GPTQ_QY parameters have output dimension on dim1.
    int fused_dim = get_fused_split_dim();
    std::vector<SplitParam> result;
    auto qw_it = params.find("qweight");
    auto qz_it = params.find("qzeros");
    auto sc_it = params.find("scales");
    auto gidx_it = params.find("g_idx");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".qweight",
                          infinicore::nn::Parameter(
                              qw_it->second->narrow({{static_cast<size_t>(fused_dim), s.start, s.size}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".qzeros",
                          infinicore::nn::Parameter(
                              qz_it->second->narrow({{static_cast<size_t>(fused_dim), s.start, s.size}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".scales",
                          infinicore::nn::Parameter(
                              sc_it->second->narrow({{static_cast<size_t>(fused_dim), s.start, s.size}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".g_idx",
                          infinicore::nn::Parameter(
                              gidx_it->second->narrow({{0, 0, gidx_it->second->size(0)}}),
                              0, 0, 1, 0)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

// ---- Conversion from GPTQ_W4A16 ----

std::shared_ptr<BaseQuantization> GPTQ_QY::convert_from_gptq(
    ParamsMap &params,
    const infinicore::Device &device,
    const nlohmann::json &quant_config) {

    auto gptq_qy = std::make_shared<GPTQ_QY>(quant_config);
    const int bits = gptq_qy->weight_bits();
    const int values_per_int32 = 32 / bits;

    const auto &original_qweight = params.at("qweight");
    const auto &original_qzeros = params.at("qzeros");
    const auto &original_scales = params.at("scales");
    const auto &g_idx = params.at("g_idx");

    {
        const auto &shape = original_qweight->shape();
        assert(shape.size() == 2);
        size_t M = shape[0], N = shape[1];

        auto weight_unpacked = unpack_int32_to_nibbles_3d_(original_qweight, bits);
        auto weight_packed = combine_nibbles_last_dim_(weight_unpacked, M, values_per_int32, N);

        size_t dimY = N;
        size_t total_bytes = M * values_per_int32 * (N / 2);
        size_t dimX = total_bytes / dimY;

        assert(dimX * dimY == total_bytes && "Weight shape calculation mismatch");

        params["qweight"] = make_tensor_from_host_(
            weight_packed.data(), total_bytes * sizeof(uint8_t),
            {dimX, dimY}, infinicore::DataType::U8, device);
    }

    {
        const auto &shape = original_qzeros->shape();
        assert(shape.size() == 2);
        size_t P = shape[0], Q = shape[1];

        auto zeros_fp32 = unpack_zeros_to_fp32_2d_(original_qzeros, bits);
        auto zeros_fp16 = infinilm::detail::float_to_fp16_bits(zeros_fp32);

        params["qzeros"] = make_tensor_from_host_(
            zeros_fp16.data(), zeros_fp16.size() * sizeof(uint16_t),
            {P, Q * static_cast<size_t>(values_per_int32)},
            infinicore::DataType::F16, device);
    }

    {
        auto scales_cpu = original_scales->to(infinicore::Device::Type::CPU);
        size_t num_elements = scales_cpu->numel();
        const void *raw_data = scales_cpu->data();

        std::vector<uint16_t> scales_fp16(num_elements);
        if (scales_cpu->dtype() == infinicore::DataType::F16) {
            std::memcpy(scales_fp16.data(), raw_data, num_elements * sizeof(uint16_t));
        } else if (scales_cpu->dtype() == infinicore::DataType::F32) {
            std::vector<float> scales_fp32(num_elements);
            std::memcpy(scales_fp32.data(), raw_data, num_elements * sizeof(float));
            scales_fp16 = infinilm::detail::float_to_fp16_bits(scales_fp32);
        } else {
            spdlog::error("Unsupported scales dtype, expected F16 or F32");
            assert(false && "Unsupported scales dtype");
        }

        params["scales"] = make_tensor_from_host_(
            scales_fp16.data(), scales_fp16.size() * sizeof(uint16_t),
            original_scales->shape(), infinicore::DataType::F16, device);
    }

    if (g_idx->numel() > 0) {
        params["g_idx"] = g_idx->to(device);
    }

    return gptq_qy;
}

// ---- Private helpers ----

std::vector<uint8_t> GPTQ_QY::unpack_int32_to_nibbles_3d_(const infinicore::Tensor &packed, int bits) {
    assert(bits == 4 || bits == 8);
    const int values_per_int32 = 32 / bits;

    auto packed_cpu = packed->to(infinicore::Device::Type::CPU);
    const int32_t *packed_host = reinterpret_cast<const int32_t *>(packed_cpu->data());

    const auto &shape = packed->shape();
    assert(shape.size() == 2);
    size_t M = shape[0], N = shape[1];

    std::vector<uint8_t> unpacked(M * values_per_int32 * N);

    for (size_t i = 0; i < M; ++i) {
        for (int k = 0; k < values_per_int32; ++k) {
            for (size_t j = 0; j < N; ++j) {
                int32_t val = packed_host[i * N + j];
                uint8_t extracted = static_cast<uint8_t>((val >> (k * bits)) & ((1 << bits) - 1));
                size_t idx = i * (values_per_int32 * N) + k * N + j;
                unpacked[idx] = extracted;
            }
        }
    }
    return unpacked;
}

std::vector<uint8_t> GPTQ_QY::combine_nibbles_last_dim_(
    const std::vector<uint8_t> &nibbles, size_t M, size_t K, size_t N) {
    assert(N % 2 == 0 && "Last dimension must be even for nibble pairing");

    std::vector<uint8_t> combined(M * K * (N / 2));
    size_t out_idx = 0;

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            size_t row_base = i * (K * N) + k * N;
            for (size_t j = 0; j < N; j += 2) {
                uint8_t low = nibbles[row_base + j] & 0x0F;
                uint8_t high = nibbles[row_base + j + 1] & 0x0F;
                combined[out_idx++] = static_cast<uint8_t>((high << 4) | low);
            }
        }
    }
    return combined;
}

std::vector<float> GPTQ_QY::unpack_zeros_to_fp32_2d_(const infinicore::Tensor &packed_zeros, int bits) {
    assert(bits == 4 || bits == 8);
    const int values_per_int32 = 32 / bits;
    const int mask = (1 << bits) - 1;

    auto packed_cpu = packed_zeros->to(infinicore::Device::Type::CPU);
    const int32_t *packed_host = reinterpret_cast<const int32_t *>(packed_cpu->data());

    const auto &shape = packed_zeros->shape();
    assert(shape.size() == 2);
    size_t P = shape[0], Q = shape[1];

    std::vector<float> result(P * Q * values_per_int32);
    size_t out_idx = 0;

    for (size_t p = 0; p < P; ++p) {
        for (size_t q = 0; q < Q; ++q) {
            int32_t val = packed_host[p * Q + q];
            for (int k = 0; k < values_per_int32; ++k) {
                uint8_t extracted = static_cast<uint8_t>((val >> (k * bits)) & mask);
                int dequant_val = (static_cast<int>(extracted) + 1) & mask;
                result[out_idx++] = static_cast<float>(dequant_val);
            }
        }
    }
    return result;
}

infinicore::Tensor GPTQ_QY::make_tensor_from_host_(const void *data, size_t bytes,
                                                   const std::vector<size_t> &shape,
                                                   infinicore::DataType dtype, const infinicore::Device &device) {
    auto tensor = infinicore::Tensor::empty(shape, dtype, infinicore::Device::Type::CPU);
    std::memcpy(reinterpret_cast<void *>(tensor->data()), data, bytes);

    if (device != infinicore::Device::Type::CPU) {
        return tensor->to(device);
    }
    return tensor;
}

} // namespace infinilm::quantization
