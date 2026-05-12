#pragma once

#include "base_quantization.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <vector>

namespace infinilm::detail {

inline uint16_t fp32_to_fp16_bits(float value) {
    union {
        float f;
        uint32_t u;
    } f2u;
    f2u.f = value;
    uint32_t x = f2u.u;

    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127;
    uint32_t mantissa = x & 0x007FFFFF;

    if (exp == 128) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00);
        }
        return static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
    }
    if (exp > 15) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    if (exp < -14) {
        if (exp < -24) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x00800000;
        uint32_t shift = -exp - 14;
        mantissa >>= shift;
        if ((mantissa & 0x1000) && ((mantissa & 0x2FFF) != 0)) {
            mantissa += 0x2000;
        }
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }

    uint32_t exp16 = static_cast<uint32_t>(exp + 15) << 10;
    uint32_t mantissa16 = mantissa >> 13;
    if ((mantissa & 0x1000) && ((mantissa & 0x2FFF) || (mantissa16 & 1))) {
        mantissa16++;
        if (mantissa16 == 0x400) {
            exp16 += 0x400;
            mantissa16 = 0;
        }
    }
    return static_cast<uint16_t>(sign | exp16 | mantissa16);
}

inline std::vector<uint16_t> float_to_fp16_bits(const std::vector<float> &values) {
    std::vector<uint16_t> result;
    result.reserve(values.size());
    for (float f : values) {
        result.push_back(fp32_to_fp16_bits(f));
    }
    return result;
}

} // namespace infinilm::detail

namespace infinilm::quantization {

class GPTQ_QY : public BaseQuantization {
public:
    explicit GPTQ_QY(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {
        int bits = weight_bits();
        if (bits != 4) {
            spdlog::warn("GPTQ_QY: bits={} not fully tested, expected 4", bits);
        }
    }

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::GPTQ_W4A16_QY;
    }

    int get_packing_num() const {
        return 32 / weight_bits();
    }

    int get_group_size() const {
        return get_or<int>("group_size", 128);
    }

    int weight_bits() const { return get_or<int>("bits", 4); }
    bool desc_act() const { return get_or<bool>("desc_act", false); }

    // Parameter layout for GPTQ_QY (already converted format)
    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    int get_fused_split_dim() const override { return 1; }

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const override;

    // Split fused linear parameters into named sub-parameters
    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;

    // Convert from GPTQ_W4A16 format and update params in-place.
    // Returns a new GPTQ_QY quantization instance. Returns nullptr if
    // the device is not QY.
    static std::shared_ptr<BaseQuantization> convert_from_gptq(
        ParamsMap &params,
        const infinicore::Device &device,
        const nlohmann::json &quant_config);

private:
    static std::vector<uint8_t> unpack_int32_to_nibbles_3d_(const infinicore::Tensor &packed, int bits);
    static std::vector<uint8_t> combine_nibbles_last_dim_(const std::vector<uint8_t> &nibbles, size_t M, size_t K, size_t N);
    static std::vector<float> unpack_zeros_to_fp32_2d_(const infinicore::Tensor &packed_zeros, int bits);
    static infinicore::Tensor make_tensor_from_host_(const void *data, size_t bytes,
                                                     const std::vector<size_t> &shape,
                                                     infinicore::DataType dtype, const infinicore::Device &device);
};

} // namespace infinilm::quantization
