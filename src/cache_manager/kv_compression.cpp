#include "kv_compression.hpp"
#include "../utils.hpp"

#include <fstream>
#include <sstream>

namespace {
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

infiniDtype_t toInfiniDtype(DTypeCode code) {
    switch (code) {
    case DTypeCode::FP16:
        return INFINI_DTYPE_F16;
    case DTypeCode::BF16:
        return INFINI_DTYPE_BF16;
    case DTypeCode::FP32:
        return INFINI_DTYPE_F32;
    default:
        return INFINI_DTYPE_INVALID;
    }
}

struct WeightMeta {
    uint32_t rows;
    uint32_t cols;
    uint32_t has_bias;
};
static_assert(sizeof(WeightMeta) == 12, "WeightMeta size mismatch");
} // namespace

Compressor::Compressor(const CompressionConfig &cfg) : config_(cfg) {}

bool Compressor::loadWeights() {
    // Guard: require config enabled and weight path set.
    if (!config_.enable) {
        return false;
    }
    if (config_.weight_path.empty()) {
        return false;
    }
    // PyTorch .pth is not supported directly.
    if (config_.weight_path.size() >= 4 &&
        config_.weight_path.substr(config_.weight_path.size() - 4) == ".pth") {
        std::stringstream ss;
        ss << "Unsupported weight format (.pth) in "
           << config_.weight_path
           << "; convert to binary format described in docs/KVCacheCompressionWeightFormat.md";
        std::cerr << ss.str() << std::endl;
        return false;
    }

    std::ifstream fin(config_.weight_path, std::ios::binary);
    if (!fin) {
        std::stringstream ss;
        ss << "Failed to open weight file: " << config_.weight_path;
        std::cerr << ss.str() << std::endl;
        return false;
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
        std::cerr << "Failed to read compression weight header" << std::endl;
        return false;
    }
    std::cerr << "Header: magic=" << std::hex << hdr.magic << std::dec
              << " version=" << hdr.version
              << " dtype_code=" << hdr.dtype_code
              << " num_layers=" << hdr.num_layers
              << " weight_count_per_layer=" << hdr.weight_count_per_layer
              << " meta_size=" << hdr.metadata_size_bytes
              << std::endl;
    if (hdr.magic != kMagic || hdr.version != 1) {
        std::cerr << "Invalid compression weight header" << std::endl;
        return false;
    }
    // Basic sanity checks on header fields.
    if (hdr.num_layers == 0 || hdr.num_layers > 10000 ||
        hdr.weight_count_per_layer == 0 || hdr.weight_count_per_layer > 4096) {
        std::cerr << "Invalid header values (num_layers/weight_count_per_layer)" << std::endl;
        return false;
    }
    auto dtype = toInfiniDtype(static_cast<DTypeCode>(hdr.dtype_code));
    if (dtype == INFINI_DTYPE_INVALID) {
        std::cerr << "Unsupported dtype in compression weight file" << std::endl;
        return false;
    }
    // Sync config with header if not set
    if (config_.compression_factor == 0 || config_.compression_factor == 1) {
        config_.compression_factor = hdr.compression_factor;
    }
    if (config_.min_seq_len == 0) {
        config_.min_seq_len = hdr.min_seq_len;
    }

    // Skip metadata if present.
    if (hdr.metadata_size_bytes > 0) {
        fin.seekg(hdr.metadata_size_bytes, std::ios::cur);
    }

    weights_.clear();
    prefix_offsets_.clear();
    layered_weights_.clear();
    weights_.reserve(static_cast<size_t>(hdr.num_layers) * hdr.weight_count_per_layer);
    layered_weights_.resize(hdr.num_layers);

    // Record layer offsets to map (layer, index) -> weights_ position.
    for (uint32_t layer = 0; layer < hdr.num_layers; ++layer) {
        prefix_offsets_.push_back(static_cast<uint32_t>(weights_.size()));
        layered_weights_[layer].reserve(hdr.weight_count_per_layer);
        for (uint32_t w = 0; w < hdr.weight_count_per_layer; ++w) {
            WeightMeta meta{};
            char meta_buf[sizeof(WeightMeta)];
            fin.read(meta_buf, sizeof(WeightMeta));
            if (!fin) {
                std::cerr << "Unexpected EOF while reading weight meta" << std::endl;
                return false;
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

            const size_t weight_elems = static_cast<size_t>(meta.rows) * meta.cols;
            // Guard against unreasonable sizes to avoid allocation overflow.
            const size_t max_elems = static_cast<size_t>(1e8); // ~200MB for fp16
            if (meta.rows == 0 || meta.cols == 0 || weight_elems > max_elems) {
                std::cerr << "Unreasonable weight shape: rows=" << meta.rows
                          << " cols=" << meta.cols << std::endl;
                return false;
            }
            const size_t weight_bytes = weight_elems * dsize(dtype);
            std::vector<char> buf(weight_bytes);
            fin.read(buf.data(), static_cast<std::streamsize>(weight_bytes));
            if (!fin) {
                std::cerr << "Unexpected EOF while reading weight data" << std::endl;
                return false;
            }

            // Create Tensor on device.
            auto weight_tensor = Tensor::weight(buf.data(), dtype, {meta.rows, meta.cols});
            weights_.push_back(weight_tensor);

            std::shared_ptr<Tensor> bias_tensor = nullptr;
            if (meta.has_bias) {
                const size_t bias_bytes = static_cast<size_t>(meta.rows) * dsize(dtype);
                std::vector<char> bias_buf(bias_bytes);
                fin.read(bias_buf.data(), static_cast<std::streamsize>(bias_bytes));
                if (!fin) {
                    std::cerr << "Unexpected EOF while reading bias" << std::endl;
                    return false;
                }
                bias_tensor = Tensor::weight(bias_buf.data(), dtype, {meta.rows});
                weights_.push_back(bias_tensor);
            }
            layered_weights_[layer].push_back(LinearWeight{weight_tensor, bias_tensor});
        }
    }

    return true;
}

std::shared_ptr<Tensor> Compressor::getWeight(uint32_t layer, uint32_t idx) const {
    if (layer >= prefix_offsets_.size()) {
        return nullptr;
    }
    uint32_t base = prefix_offsets_[layer];
    uint32_t pos = base + idx;
    if (pos >= weights_.size()) {
        return nullptr;
    }
    return weights_[pos];
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
Compressor::getLinearWithBias(uint32_t layer, uint32_t prefix_idx, uint32_t slot) const {
    // prefix_idx in [0, kPrefixCount), slot in [0, kWeightsPerPrefix)
    const uint32_t idx = prefix_idx * kWeightsPerPrefix + slot;
    if (layer >= layered_weights_.size()) {
        return {nullptr, nullptr};
    }
    const auto &vec = layered_weights_[layer];
    if (idx >= vec.size()) {
        return {nullptr, nullptr};
    }
    return {vec[idx].weight, vec[idx].bias};
}
