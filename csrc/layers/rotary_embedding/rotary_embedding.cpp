#include "rotary_embedding.hpp"

#include "../../backends/operators/operators.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infinilm::layers::rotary_embedding {
namespace {
thread_local std::unordered_map<std::string, std::shared_ptr<infinicore::nn::RoPE>> _ROPE_DICT;
thread_local std::unordered_map<std::string, std::shared_ptr<RotaryEmbedding>> _ROTARY_DICT;

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t f32;
    std::memcpy(&f32, &value, sizeof(f32));
    const std::uint16_t sign = (f32 >> 16) & 0x8000;
    const std::int32_t exponent = static_cast<std::int32_t>((f32 >> 23) & 0xFF) - 127;
    std::uint32_t mantissa = f32 & 0x7FFFFF;

    if (exponent >= 16) {
        if (exponent == 128 && mantissa != 0) {
            return static_cast<std::uint16_t>(sign | 0x7E00);
        }
        return static_cast<std::uint16_t>(sign | 0x7C00);
    }
    if (exponent >= -14) {
        return static_cast<std::uint16_t>(
            sign | ((exponent + 15) << 10) | (mantissa >> 13));
    }
    if (exponent >= -24) {
        mantissa |= 0x800000;
        mantissa >>= (-14 - exponent);
        return static_cast<std::uint16_t>(sign | (mantissa >> 13));
    }
    return sign;
}

std::uint16_t f32_to_bf16_bits(float value) {
    std::uint32_t bits32;
    std::memcpy(&bits32, &value, sizeof(bits32));
    const std::uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    return static_cast<std::uint16_t>((bits32 + rounding_bias) >> 16);
}

std::string make_cache_key(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                           const infinicore::Device &device) {
    std::ostringstream os;
    os << device.toString()
       << "|dtype=" << static_cast<int>(model_config->get_dtype())
       << "|head_dim=" << model_config->get_head_dim()
       << "|max_pos=" << model_config->get<size_t>("max_position_embeddings")
       << std::setprecision(17)
       << "|theta=" << model_config->get<double>("rope_theta");

    const auto &json = model_config->get_config_json();
    if (json.contains("rope_scaling") && !json["rope_scaling"].is_null()) {
        os << "|scaling=" << json["rope_scaling"].dump();
    } else {
        os << "|scaling=null";
    }

    return os.str();
}

} // namespace

RotaryEmbedding::RotaryEmbedding(size_t head_dim,
                                 size_t max_seq_len,
                                 double theta,
                                 const infinicore::DataType &dtype,
                                 const infinicore::Device &device,
                                 std::shared_ptr<infinicore::nn::RoPE::ScalingConfig> scaling)
    : head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      theta_(theta),
      dtype_(dtype),
      device_(device),
      scaling_(std::move(scaling)),
      legacy_(std::make_shared<infinicore::nn::RoPE>(
          head_dim_, max_seq_len_, theta_, infinicore::nn::RoPE::Algo::GPT_NEOX,
          dtype_, device_, scaling_)) {
    if (infinilm::backends::ops::should_use(device_)) {
        initialize_cos_sin_cache();
    }
}

void RotaryEmbedding::initialize_cos_sin_cache() {
    const size_t cache_dim = head_dim_ / 2;
    std::vector<float> cos_sin_f32(max_seq_len_ * head_dim_);

    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        for (size_t j = 0; j < cache_dim; ++j) {
            float table_factor = 1.0f;
            float inv_freq;
            if (scaling_ == nullptr) {
                inv_freq = 1.0f / std::pow(static_cast<float>(theta_), 2.0f * static_cast<float>(j) / static_cast<float>(head_dim_));
            } else if (scaling_->type() == infinicore::nn::RoPE::ScalingType::LONGROPE) {
                auto lr = std::dynamic_pointer_cast<infinicore::nn::RoPE::LongRopeConfig>(scaling_);
                table_factor = lr->factor();
                const float ext = pos < lr->original_max_position_embeddings()
                                    ? lr->short_factor()[j]
                                    : lr->long_factor()[j];
                inv_freq = 1.0f / (ext * std::pow(static_cast<float>(theta_), 2.0f * static_cast<float>(j) / static_cast<float>(head_dim_)));
            } else {
                inv_freq = 1.0f / std::pow(static_cast<float>(theta_), 2.0f * static_cast<float>(j) / static_cast<float>(head_dim_));
            }

            const float angle = static_cast<float>(pos) * inv_freq;
            cos_sin_f32[pos * head_dim_ + j] = std::cos(angle) * table_factor;
            cos_sin_f32[pos * head_dim_ + cache_dim + j] = std::sin(angle) * table_factor;
        }
    }

    auto cpu = infinicore::Device::cpu();
    cos_sin_cache_ = infinicore::Tensor::empty({max_seq_len_, head_dim_}, dtype_, device_);
    if (dtype_ == infinicore::DataType::F32) {
        auto cache_cpu = infinicore::Tensor::from_blob(cos_sin_f32.data(), {max_seq_len_, head_dim_}, dtype_, cpu);
        cos_sin_cache_->copy_from(cache_cpu);
    } else if (dtype_ == infinicore::DataType::BF16) {
        std::vector<std::uint16_t> cos_sin_bf16(cos_sin_f32.size());
        for (size_t i = 0; i < cos_sin_f32.size(); ++i) {
            cos_sin_bf16[i] = f32_to_bf16_bits(cos_sin_f32[i]);
        }
        auto cache_cpu = infinicore::Tensor::from_blob(cos_sin_bf16.data(), {max_seq_len_, head_dim_}, dtype_, cpu);
        cos_sin_cache_->copy_from(cache_cpu);
    } else if (dtype_ == infinicore::DataType::F16) {
        std::vector<std::uint16_t> cos_sin_f16(cos_sin_f32.size());
        for (size_t i = 0; i < cos_sin_f32.size(); ++i) {
            cos_sin_f16[i] = f32_to_f16_bits(cos_sin_f32[i]);
        }
        auto cache_cpu = infinicore::Tensor::from_blob(cos_sin_f16.data(), {max_seq_len_, head_dim_}, dtype_, cpu);
        cos_sin_cache_->copy_from(cache_cpu);
    } else {
        throw std::runtime_error("RotaryEmbedding: unsupported cache dtype for InfiniOps adapter");
    }
}

bool RotaryEmbedding::try_infiniops(const infinicore::Tensor &query_out,
                                    const infinicore::Tensor &query,
                                    const infinicore::Tensor &key_out,
                                    const infinicore::Tensor &key,
                                    const infinicore::Tensor &positions) const {
    if (!infinilm::backends::ops::should_use(query->device())) {
        return false;
    }
    if (!cos_sin_cache_) {
        throw std::runtime_error("RotaryEmbedding: InfiniOps cache was not initialized");
    }

    infinicore::Tensor q = query;
    infinicore::Tensor k = key;
    infinicore::Tensor q_out = query_out;
    infinicore::Tensor k_out = key_out;

    if (query->ndim() == 4) {
        if (query->size(0) != 1 || key->size(0) != 1) {
            throw std::runtime_error("RotaryEmbedding: InfiniOps adapter only supports 4D RoPE when batch size is 1");
        }
        q = query->view({query->size(1), query->size(2), query->size(3)});
        k = key->view({key->size(1), key->size(2), key->size(3)});
        q_out = query_out->view({query_out->size(1), query_out->size(2), query_out->size(3)});
        k_out = key_out->view({key_out->size(1), key_out->size(2), key_out->size(3)});
    } else if (query->ndim() != 3) {
        throw std::runtime_error("RotaryEmbedding: InfiniOps adapter expects 3D or 4D query tensors");
    }

    if (!q->is_contiguous() || !k->is_contiguous() || !q_out->is_contiguous() || !k_out->is_contiguous()) {
        throw std::runtime_error("RotaryEmbedding: InfiniOps adapter requires contiguous Q/K tensors");
    }
    if (positions->numel() != q->size(0)) {
        throw std::runtime_error("RotaryEmbedding: positions length must match the flattened token count");
    }

    if (!positions->is_contiguous()) {
        throw std::runtime_error("RotaryEmbedding: InfiniOps adapter requires contiguous positions");
    }
    infinilm::backends::ops::rotary_embedding(
        positions, q, k, cos_sin_cache_, static_cast<int64_t>(head_dim_), q_out, k_out);
    return true;
}

void RotaryEmbedding::forward_pair(const infinicore::Tensor &query_out,
                                   const infinicore::Tensor &query,
                                   const infinicore::Tensor &key_out,
                                   const infinicore::Tensor &key,
                                   const infinicore::Tensor &positions) const {
    if (try_infiniops(query_out, query, key_out, key, positions)) {
        return;
    }

    legacy_->forward(query_out, query, positions);
    if (key_out->data() == key->data()) {
        legacy_->forward(key, positions, true);
    } else {
        legacy_->forward(key_out, key, positions);
    }
}

void RotaryEmbedding::forward_pair_inplace(infinicore::Tensor &query,
                                           infinicore::Tensor &key,
                                           const infinicore::Tensor &positions) const {
    if (try_infiniops(query, query, key, key, positions)) {
        return;
    }

    legacy_->forward(query, positions, true);
    legacy_->forward(key, positions, true);
}

std::shared_ptr<RotaryEmbedding> get_rotary_embedding(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                                      const infinicore::Device &device) {
    const auto key = make_cache_key(model_config, device);
    auto it = _ROTARY_DICT.find(key);
    if (it != _ROTARY_DICT.end()) {
        return it->second;
    }

    auto rope = std::make_shared<RotaryEmbedding>(
        model_config->get_head_dim(),
        model_config->get<size_t>("max_position_embeddings"),
        model_config->get<double>("rope_theta"),
        model_config->get_dtype(),
        device,
        model_config->get_rope_scaling());
    _ROTARY_DICT.emplace(key, rope);
    return rope;
}

std::shared_ptr<infinicore::nn::RoPE> get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                               const infinicore::Device &device) {
    const std::string scaling_type = "default";
    auto it = _ROPE_DICT.find(scaling_type);
    if (it != _ROPE_DICT.end()) {
        return it->second;
    }

    const auto &dtype = model_config->get_dtype();
    size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    double rope_theta = model_config->get<double>("rope_theta");
    auto rope = std::make_shared<infinicore::nn::RoPE>(model_config->get_head_dim(), max_position_embeddings, rope_theta,
                                                       infinicore::nn::RoPE::Algo::GPT_NEOX, dtype, device,
                                                       model_config->get_rope_scaling());

    _ROPE_DICT.emplace(scaling_type, rope);
    return rope;
}

} // namespace infinilm::layers::rotary_embedding
