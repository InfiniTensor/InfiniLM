#include "kv_compression.hpp"

#include "../utils.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace infinilm::cache {
namespace kv_compression_detail {
// Magic: "KV C M" little endian = 0x4b56434d
constexpr uint32_t kMagic = 0x4b56434d;

struct Header {
    uint32_t magic;
    uint32_t version;
    uint16_t dtype_code;
    uint16_t reserved;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t hidden_size;
    uint32_t compression_factor;
    uint32_t min_seq_len;
    uint32_t weight_count_per_layer;
    uint32_t metadata_size_bytes;
};
static_assert(sizeof(Header) == 44, "Header size mismatch");

enum class DTypeCode : uint16_t { FP16 = 0, BF16 = 1, FP32 = 2 };

infinicore::DataType toDataType(DTypeCode code) {
    switch (code) {
    case DTypeCode::FP16:
        return infinicore::DataType::F16;
    case DTypeCode::BF16:
        return infinicore::DataType::BF16;
    case DTypeCode::FP32:
        return infinicore::DataType::F32;
    default:
        return infinicore::DataType::F16;
    }
}

struct WeightMeta {
    uint32_t rows;
    uint32_t cols;
    uint32_t has_bias;
};
static_assert(sizeof(WeightMeta) == 12, "WeightMeta size mismatch");

infinicore::Tensor cast_tensor_cpu(const infinicore::Tensor &src,
                                   infinicore::DataType target_dtype) {
    if (!src) {
        return {};
    }
    if (src->dtype() == target_dtype) {
        return src;
    }
    if (src->device().getType() != infinicore::Device::Type::CPU) {
        throw std::runtime_error("cast_tensor_cpu: source tensor must be on CPU");
    }

    auto dst = infinicore::Tensor::empty(src->shape(), target_dtype, infinicore::Device::cpu());
    auto n = src->numel();
    auto *src_ptr = src->data();
    auto *dst_ptr = dst->data();

    auto read_f32 = [&](size_t i) -> float {
        switch (src->dtype()) {
        case infinicore::DataType::F16:
            return f16_to_f32(reinterpret_cast<const uint16_t *>(src_ptr)[i]);
        case infinicore::DataType::BF16:
            return bf16_to_f32(reinterpret_cast<const uint16_t *>(src_ptr)[i]);
        case infinicore::DataType::F32:
            return reinterpret_cast<const float *>(src_ptr)[i];
        default:
            throw std::runtime_error("cast_tensor_cpu: unsupported source dtype");
        }
    };

    if (target_dtype == infinicore::DataType::F16) {
        auto *out = reinterpret_cast<uint16_t *>(dst_ptr);
        for (size_t i = 0; i < n; ++i) {
            out[i] = f32_to_f16(read_f32(i));
        }
    } else if (target_dtype == infinicore::DataType::BF16) {
        auto *out = reinterpret_cast<uint16_t *>(dst_ptr);
        for (size_t i = 0; i < n; ++i) {
            out[i] = f32_to_bf16(read_f32(i));
        }
    } else if (target_dtype == infinicore::DataType::F32) {
        auto *out = reinterpret_cast<float *>(dst_ptr);
        for (size_t i = 0; i < n; ++i) {
            out[i] = read_f32(i);
        }
    } else {
        throw std::runtime_error("cast_tensor_cpu: unsupported target dtype");
    }
    return dst;
}

std::string make_device_key(const infinicore::Device &device, infinicore::DataType dtype) {
    return device.toString() + ":" + std::to_string(static_cast<int>(dtype));
}

class KVCompressor {
public:
    explicit KVCompressor(KVCompressionConfig cfg) : config_(std::move(cfg)) {}

    bool load_weights() {
        if (!config_.enable || config_.weight_path.empty()) {
            return false;
        }
        if (config_.weight_path.size() >= 4 &&
            config_.weight_path.substr(config_.weight_path.size() - 4) == ".pth") {
            std::stringstream ss;
            ss << "Unsupported weight format (.pth) in " << config_.weight_path
               << "; convert to binary format described in docs/KVCacheCompressionWeightFormat.md";
            throw std::runtime_error(ss.str());
        }

        std::ifstream fin(config_.weight_path, std::ios::binary);
        if (!fin) {
            std::stringstream ss;
            ss << "Failed to open weight file: " << config_.weight_path;
            throw std::runtime_error(ss.str());
        }

        auto read_u32 = [&](uint32_t &out) -> bool {
            char buf[4];
            fin.read(buf, 4);
            if (!fin) return false;
            out = static_cast<uint32_t>(static_cast<unsigned char>(buf[0])) |
                  (static_cast<uint32_t>(static_cast<unsigned char>(buf[1])) << 8) |
                  (static_cast<uint32_t>(static_cast<unsigned char>(buf[2])) << 16) |
                  (static_cast<uint32_t>(static_cast<unsigned char>(buf[3])) << 24);
            return true;
        };
        auto read_u16 = [&](uint16_t &out) -> bool {
            char buf[2];
            fin.read(buf, 2);
            if (!fin) return false;
            out = static_cast<uint16_t>(static_cast<unsigned char>(buf[0])) |
                  (static_cast<uint16_t>(static_cast<unsigned char>(buf[1])) << 8);
            return true;
        };

        Header hdr{};
        if (!read_u32(hdr.magic) || !read_u32(hdr.version) ||
            !read_u16(hdr.dtype_code) || !read_u16(hdr.reserved) ||
            !read_u32(hdr.num_layers) || !read_u32(hdr.num_heads) ||
            !read_u32(hdr.head_dim) || !read_u32(hdr.hidden_size) ||
            !read_u32(hdr.compression_factor) || !read_u32(hdr.min_seq_len) ||
            !read_u32(hdr.weight_count_per_layer) || !read_u32(hdr.metadata_size_bytes)) {
            throw std::runtime_error("Failed to read compression weight header");
        }
        if (hdr.magic != kMagic || hdr.version != 1) {
            throw std::runtime_error("Invalid compression weight header");
        }
        if (hdr.num_layers == 0 || hdr.weight_count_per_layer == 0) {
            throw std::runtime_error("Invalid header values (num_layers/weight_count_per_layer)");
        }

        weight_dtype_ = toDataType(static_cast<DTypeCode>(hdr.dtype_code));
        if (config_.compression_factor <= 1) {
            config_.compression_factor = hdr.compression_factor;
        }
        if (config_.min_seq_len == 0) {
            config_.min_seq_len = hdr.min_seq_len;
        }

        if (hdr.metadata_size_bytes > 0) {
            fin.seekg(hdr.metadata_size_bytes, std::ios::cur);
        }

        layered_weights_cpu_.clear();
        layered_weights_cpu_.resize(hdr.num_layers);

        for (uint32_t layer = 0; layer < hdr.num_layers; ++layer) {
            layered_weights_cpu_[layer].reserve(hdr.weight_count_per_layer);
            for (uint32_t w = 0; w < hdr.weight_count_per_layer; ++w) {
                WeightMeta meta{};
                char meta_buf[sizeof(WeightMeta)];
                fin.read(meta_buf, sizeof(WeightMeta));
                if (!fin) {
                    throw std::runtime_error("Unexpected EOF while reading weight meta");
                }
                meta.rows = static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[0])) |
                            (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[1])) << 8) |
                            (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[2])) << 16) |
                            (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[3])) << 24);
                meta.cols = static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[4])) |
                            (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[5])) << 8) |
                            (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[6])) << 16) |
                            (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[7])) << 24);
                meta.has_bias = static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[8])) |
                                (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[9])) << 8) |
                                (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[10])) << 16) |
                                (static_cast<uint32_t>(static_cast<unsigned char>(meta_buf[11])) << 24);

                if (meta.rows == 0 || meta.cols == 0) {
                    throw std::runtime_error("Invalid weight shape");
                }
                size_t weight_elems = static_cast<size_t>(meta.rows) * meta.cols;
                size_t weight_bytes = weight_elems * infinicore::dsize(weight_dtype_);
                std::vector<char> buf(weight_bytes);
                fin.read(buf.data(), static_cast<std::streamsize>(weight_bytes));
                if (!fin) {
                    throw std::runtime_error("Unexpected EOF while reading weight data");
                }

                auto weight_tensor = infinicore::Tensor::empty(
                    {meta.rows, meta.cols}, weight_dtype_, infinicore::Device::cpu());
                std::memcpy(weight_tensor->data(), buf.data(), weight_bytes);

                std::optional<infinicore::Tensor> bias_tensor;
                if (meta.has_bias) {
                    size_t bias_bytes = static_cast<size_t>(meta.rows) * infinicore::dsize(weight_dtype_);
                    std::vector<char> bias_buf(bias_bytes);
                    fin.read(bias_buf.data(), static_cast<std::streamsize>(bias_bytes));
                    if (!fin) {
                        throw std::runtime_error("Unexpected EOF while reading bias");
                    }
                    auto bias = infinicore::Tensor::empty({meta.rows}, weight_dtype_, infinicore::Device::cpu());
                    std::memcpy(bias->data(), bias_buf.data(), bias_bytes);
                    bias_tensor = bias;
                }
                layered_weights_cpu_[layer].push_back(LinearWeight{weight_tensor, bias_tensor});
            }
        }
        return true;
    }

    uint32_t compress_inplace(StaticKVCache &cache, uint32_t seq_len, size_t batch_size) {
        if (!config_.enable || seq_len == 0 || config_.compression_factor <= 1) {
            return seq_len;
        }
        if (!cache.k_caches_ || !cache.v_caches_) {
            return seq_len;
        }
        if (batch_size == 0) {
            return seq_len;
        }
        seq_len = std::min<uint32_t>(seq_len, static_cast<uint32_t>(cache.cache_len_));
        batch_size = std::min(batch_size, static_cast<size_t>(cache.rank_batch_size_));

        auto device = cache.k_caches_->device();
        auto dtype = cache.k_caches_->dtype();
        const auto &layered_weights = get_weights_for_device(device, dtype);
        if (layered_weights.empty()) {
            return seq_len;
        }

        const uint32_t factor = config_.compression_factor;
        const uint32_t min_seq_len = config_.min_seq_len;
        const uint32_t img_len = std::min<uint32_t>(config_.image_kv_len, seq_len);
        const uint32_t txt_len = seq_len - img_len;

        const bool has_text_mlp = has_prefix_mlp(layered_weights, 0) && has_prefix_mlp(layered_weights, 1);
        const bool has_image_mlp = has_prefix_mlp(layered_weights, 2) && has_prefix_mlp(layered_weights, 3);
        if (!has_text_mlp) {
            return seq_len;
        }

        uint32_t new_len = 0;

        for (size_t layer = 0; layer < cache.rank_num_layers_; ++layer) {
            auto k_layer = cache.k_caches_->narrow({{0, layer, 1}})->squeeze(0);
            auto v_layer = cache.v_caches_->narrow({{0, layer, 1}})->squeeze(0);

            for (size_t b = 0; b < batch_size; ++b) {
                auto k_batch = k_layer->narrow({{0, b, 1}})->squeeze(0);
                auto v_batch = v_layer->narrow({{0, b, 1}})->squeeze(0);

                auto k_view = k_batch->narrow({{1, 0, seq_len}});
                auto v_view = v_batch->narrow({{1, 0, seq_len}});

                auto k_img = img_len > 0 ? k_view->narrow({{1, 0, img_len}}) : infinicore::Tensor{};
                auto v_img = img_len > 0 ? v_view->narrow({{1, 0, img_len}}) : infinicore::Tensor{};
                auto k_txt = txt_len > 0 ? k_view->narrow({{1, img_len, txt_len}}) : infinicore::Tensor{};
                auto v_txt = txt_len > 0 ? v_view->narrow({{1, img_len, txt_len}}) : infinicore::Tensor{};

                infinicore::Tensor k_img_comp;
                infinicore::Tensor v_img_comp;
                infinicore::Tensor k_txt_comp;
                infinicore::Tensor v_txt_comp;

                if (img_len > 0) {
                    if (has_image_mlp) {
                        auto res = compress_segment(layered_weights, layer, k_img, v_img, factor, min_seq_len, 2);
                        k_img_comp = res.first;
                        v_img_comp = res.second;
                    } else {
                        k_img_comp = k_img;
                        v_img_comp = v_img;
                    }
                }
                if (txt_len > 0) {
                    auto res = compress_segment(layered_weights, layer, k_txt, v_txt, factor, min_seq_len, 0);
                    k_txt_comp = res.first;
                    v_txt_comp = res.second;
                }

                infinicore::Tensor k_comp;
                infinicore::Tensor v_comp;

                if (k_img_comp && k_txt_comp) {
                    auto total_len = k_img_comp->size(1) + k_txt_comp->size(1);
                    k_comp = infinicore::Tensor::empty({k_img_comp->size(0), total_len, k_img_comp->size(2)},
                                                       k_img_comp->dtype(), k_img_comp->device());
                    v_comp = infinicore::Tensor::empty({v_img_comp->size(0), total_len, v_img_comp->size(2)},
                                                       v_img_comp->dtype(), v_img_comp->device());

                    k_comp->narrow({{1, 0, k_img_comp->size(1)}})->copy_from(k_img_comp);
                    k_comp->narrow({{1, k_img_comp->size(1), k_txt_comp->size(1)}})->copy_from(k_txt_comp);
                    v_comp->narrow({{1, 0, v_img_comp->size(1)}})->copy_from(v_img_comp);
                    v_comp->narrow({{1, v_img_comp->size(1), v_txt_comp->size(1)}})->copy_from(v_txt_comp);
                } else {
                    k_comp = k_img_comp ? k_img_comp : k_txt_comp;
                    v_comp = v_img_comp ? v_img_comp : v_txt_comp;
                }

                if (!k_comp || !v_comp) {
                    return seq_len;
                }

                uint32_t comp_len = static_cast<uint32_t>(k_comp->size(1));
                if (new_len == 0) {
                    new_len = comp_len;
                } else if (new_len != comp_len) {
                    return seq_len;
                }

                k_batch->narrow({{1, 0, comp_len}})->copy_from(k_comp);
                v_batch->narrow({{1, 0, comp_len}})->copy_from(v_comp);
            }
        }

        infinicore::context::syncDevice();
        return new_len > 0 ? new_len : seq_len;
    }

private:
    struct LinearWeight {
        infinicore::Tensor weight;
        std::optional<infinicore::Tensor> bias;
    };

    KVCompressionConfig config_;
    infinicore::DataType weight_dtype_{infinicore::DataType::F16};
    std::vector<std::vector<LinearWeight>> layered_weights_cpu_;
    std::unordered_map<std::string, std::vector<std::vector<LinearWeight>>> weights_cache_;
    std::mutex weights_mutex_;

    static constexpr uint32_t kWeightsPerPrefix = 3;
    static constexpr uint32_t kPrefixCount = 4;

    const std::vector<std::vector<LinearWeight>> &get_weights_for_device(
        const infinicore::Device &device,
        infinicore::DataType target_dtype) {
        std::lock_guard<std::mutex> lock(weights_mutex_);
        auto key = make_device_key(device, target_dtype);
        auto it = weights_cache_.find(key);
        if (it != weights_cache_.end()) {
            return it->second;
        }

        std::vector<std::vector<LinearWeight>> layered;
        layered.resize(layered_weights_cpu_.size());
        for (size_t layer = 0; layer < layered_weights_cpu_.size(); ++layer) {
            layered[layer].reserve(layered_weights_cpu_[layer].size());
            for (const auto &lw : layered_weights_cpu_[layer]) {
                auto w_cpu = cast_tensor_cpu(lw.weight, target_dtype);
                auto w = device.getType() == infinicore::Device::Type::CPU ? w_cpu : w_cpu->to(device);

                std::optional<infinicore::Tensor> b;
                if (lw.bias.has_value()) {
                    auto b_cpu = cast_tensor_cpu(lw.bias.value(), target_dtype);
                    b = device.getType() == infinicore::Device::Type::CPU ? b_cpu : b_cpu->to(device);
                }
                layered[layer].push_back(LinearWeight{w, b});
            }
        }
        auto inserted = weights_cache_.emplace(key, std::move(layered));
        return inserted.first->second;
    }

    static bool has_prefix_mlp(const std::vector<std::vector<LinearWeight>> &layered_weights,
                               uint32_t prefix_idx) {
        if (layered_weights.empty()) {
            return false;
        }
        for (uint32_t slot = 0; slot < kWeightsPerPrefix; ++slot) {
            uint32_t idx = prefix_idx * kWeightsPerPrefix + slot;
            if (idx >= layered_weights[0].size()) {
                return false;
            }
            if (!layered_weights[0][idx].weight) {
                return false;
            }
        }
        return true;
    }

    static infinicore::Tensor run_pipeline(const std::vector<std::vector<LinearWeight>> &layered_weights,
                                           size_t layer,
                                           const infinicore::Tensor &input2d,
                                           uint32_t prefix_idx) {
        if (layer >= layered_weights.size()) {
            return {};
        }
        const auto &weights = layered_weights[layer];
        const uint32_t base = prefix_idx * kWeightsPerPrefix;
        if (base + 2 >= weights.size()) {
            return {};
        }
        const auto &l0 = weights[base];
        const auto &l1 = weights[base + 1];
        const auto &l2 = weights[base + 2];
        if (!l0.weight || !l1.weight || !l2.weight) {
            return {};
        }

        if (l0.weight->size(1) != input2d->size(1)) {
            return {};
        }
        if (l1.weight->size(1) != l0.weight->size(0) ||
            l2.weight->size(1) != l1.weight->size(0)) {
            return {};
        }

        auto out0 = infinicore::op::linear(input2d, l0.weight, l0.bias);
        out0 = infinicore::op::relu(out0);
        auto out1 = infinicore::op::linear(out0, l1.weight, l1.bias);
        out1 = infinicore::op::relu(out1);
        auto out2 = infinicore::op::linear(out1, l2.weight, l2.bias);
        return out2;
    }

    static std::pair<infinicore::Tensor, infinicore::Tensor> compress_segment(
        const std::vector<std::vector<LinearWeight>> &layered_weights,
        size_t layer,
        const infinicore::Tensor &k_seg,
        const infinicore::Tensor &v_seg,
        uint32_t factor,
        uint32_t min_seq_len,
        uint32_t prefix_base) {
        if (!k_seg || !v_seg) {
            return {infinicore::Tensor{}, infinicore::Tensor{}};
        }

        const uint32_t nkvh = static_cast<uint32_t>(k_seg->size(0));
        const uint32_t seg_len = static_cast<uint32_t>(k_seg->size(1));
        const uint32_t dk = static_cast<uint32_t>(k_seg->size(2));

        uint32_t compressed_seq_len = seg_len / factor;
        if (compressed_seq_len < min_seq_len) {
            return {k_seg, v_seg};
        }
        uint32_t compress_len = compressed_seq_len * factor;
        uint32_t remainder_len = seg_len - compress_len;

        auto k_head = k_seg->narrow({{1, 0, compress_len}})->contiguous();
        auto v_head = v_seg->narrow({{1, 0, compress_len}})->contiguous();

        auto k_grouped = k_head->view({nkvh, compress_len / factor, factor * dk});
        auto v_grouped = v_head->view({nkvh, compress_len / factor, factor * dk});

        const size_t rows_linear = static_cast<size_t>(compress_len / factor) * nkvh;
        auto k_in2d = k_grouped->view({rows_linear, factor * dk});
        auto v_in2d = v_grouped->view({rows_linear, factor * dk});

        auto k_comp2d = run_pipeline(layered_weights, layer, k_in2d, prefix_base);
        auto v_comp2d = run_pipeline(layered_weights, layer, v_in2d, prefix_base + 1);
        if (!k_comp2d || !v_comp2d) {
            return {infinicore::Tensor{}, infinicore::Tensor{}};
        }

        auto k_comp_head = k_comp2d->view({nkvh, compress_len / factor, dk});
        auto v_comp_head = v_comp2d->view({nkvh, compress_len / factor, dk});

        if (remainder_len == 0) {
            return {k_comp_head, v_comp_head};
        }

        auto k_comp = infinicore::Tensor::empty({nkvh, compressed_seq_len + remainder_len, dk},
                                                k_seg->dtype(), k_seg->device());
        auto v_comp = infinicore::Tensor::empty({nkvh, compressed_seq_len + remainder_len, dk},
                                                v_seg->dtype(), v_seg->device());

        k_comp->narrow({{1, 0, compressed_seq_len}})->copy_from(k_comp_head);
        v_comp->narrow({{1, 0, compressed_seq_len}})->copy_from(v_comp_head);

        auto k_tail = k_seg->narrow({{1, compress_len, remainder_len}});
        auto v_tail = v_seg->narrow({{1, compress_len, remainder_len}});

        k_comp->narrow({{1, compressed_seq_len, remainder_len}})->copy_from(k_tail);
        v_comp->narrow({{1, compressed_seq_len, remainder_len}})->copy_from(v_tail);

        return {k_comp, v_comp};
    }
};

struct CompressorCacheKey {
    std::string weight_path;
    uint32_t compression_factor = 0;
    uint32_t min_seq_len = 0;
    uint32_t image_kv_len = 0;

    bool operator==(const CompressorCacheKey &o) const {
        return weight_path == o.weight_path &&
               compression_factor == o.compression_factor &&
               min_seq_len == o.min_seq_len &&
               image_kv_len == o.image_kv_len;
    }
};

struct CompressorCacheKeyHash {
    size_t operator()(const CompressorCacheKey &k) const noexcept {
        size_t h = std::hash<std::string>{}(k.weight_path);
        auto mix = [&](size_t x) {
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(std::hash<uint32_t>{}(k.compression_factor));
        mix(std::hash<uint32_t>{}(k.min_seq_len));
        mix(std::hash<uint32_t>{}(k.image_kv_len));
        return h;
    }
};

std::mutex &compressor_cache_mutex() {
    static auto *mtx = new std::mutex();
    return *mtx;
}

std::unordered_map<CompressorCacheKey, std::shared_ptr<KVCompressor>, CompressorCacheKeyHash> &compressor_cache() {
    static auto *m = new std::unordered_map<CompressorCacheKey, std::shared_ptr<KVCompressor>, CompressorCacheKeyHash>();
    return *m;
}
} // namespace kv_compression_detail

uint32_t compress_kv_cache_inplace(StaticKVCache &cache,
                                   uint32_t seq_len,
                                   size_t batch_size,
                                   const KVCompressionConfig &cfg) {
    if (!cfg.enable || seq_len == 0 || cfg.weight_path.empty()) {
        return seq_len;
    }
    if (!cache.k_caches_ || cache.k_caches_->shape().empty()) {
        return seq_len;
    }

    KVCompressionConfig cxx_cfg = cfg;

    kv_compression_detail::CompressorCacheKey key;
    key.weight_path = cxx_cfg.weight_path;
    key.compression_factor = cxx_cfg.compression_factor;
    key.min_seq_len = cxx_cfg.min_seq_len;
    key.image_kv_len = cxx_cfg.image_kv_len;

    std::shared_ptr<kv_compression_detail::KVCompressor> compressor;
    {
        std::lock_guard<std::mutex> lock(kv_compression_detail::compressor_cache_mutex());
        auto &m = kv_compression_detail::compressor_cache();
        auto it = m.find(key);
        if (it != m.end()) {
            compressor = it->second;
        } else {
            compressor = std::make_shared<kv_compression_detail::KVCompressor>(cxx_cfg);
            if (!compressor->load_weights()) {
                return seq_len;
            }
            m.emplace(key, compressor);
        }
    }

    return compressor->compress_inplace(cache, seq_len, batch_size);
}

} // namespace infinilm::cache
