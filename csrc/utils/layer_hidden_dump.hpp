#pragma once

#include "../global_state/global_state.hpp"
#include "../global_state/parallel_state.hpp"
#include "../utils.hpp"

#include "../engine/rank_barrier.hpp"
#include "infinicore/context/context.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace infinilm::utils {

inline const char *layer_hidden_dump_dir() {
    return std::getenv("INFINI_DUMP_LAYER_HIDDEN_DIR");
}

inline bool layer_hidden_dump_enabled() {
    const char *dir = layer_hidden_dump_dir();
    return dir != nullptr && dir[0] != '\0';
}

inline size_t layer_hidden_dump_row() {
    static size_t cached = static_cast<size_t>(-1);
    if (cached == static_cast<size_t>(-1)) {
        const char *raw = std::getenv("INFINI_DUMP_LAYER_HIDDEN_ROW");
        cached = (raw != nullptr && raw[0] != '\0')
                     ? static_cast<size_t>(std::strtoul(raw, nullptr, 10))
                     : 511;
    }
    return cached;
}

inline std::string layer_hidden_dump_tag() {
    const char *raw = std::getenv("INFINI_DUMP_LAYER_HIDDEN_TAG");
    return (raw != nullptr && raw[0] != '\0') ? std::string(raw) : "unknown";
}

/// When set, segment dumps (not full-layer dumps) are limited to this layer index.
/// Embed dumps are always emitted regardless of this gate.
inline bool layer_hidden_dump_segment_layer_matches(size_t layer_idx) {
    static int cached = -2;
    if (cached == -2) {
        const char *raw = std::getenv("INFINI_DUMP_LAYER_HIDDEN_LAYER");
        cached = (raw != nullptr && raw[0] != '\0')
                       ? static_cast<int>(std::strtoul(raw, nullptr, 10))
                       : -1;
    }
    if (cached < 0) {
        return true;
    }
    return layer_idx == static_cast<size_t>(cached);
}

inline std::vector<float> tensor_to_f32_vector(const infinicore::Tensor &tensor) {
    auto cpu_tensor = tensor->to(infinicore::Device(infinicore::Device::Type::CPU, 0));
    const auto dtype = cpu_tensor->dtype();
    const size_t numel = cpu_tensor->numel();
    std::vector<float> out(numel);
    std::byte *raw = cpu_tensor->data();
    if (dtype == infinicore::DataType::F32) {
        std::memcpy(out.data(), raw, numel * sizeof(float));
    } else if (dtype == infinicore::DataType::BF16) {
        auto *data = reinterpret_cast<uint16_t *>(raw);
        for (size_t i = 0; i < numel; ++i) {
            out[i] = bf16_to_f32(data[i]);
        }
    } else if (dtype == infinicore::DataType::F16) {
        auto *data = reinterpret_cast<uint16_t *>(raw);
        for (size_t i = 0; i < numel; ++i) {
            out[i] = f16_to_f32(data[i]);
        }
    } else {
        throw std::runtime_error("layer_hidden_dump: unsupported dtype");
    }
    return out;
}

/// Synchronize all TP worker threads before rank-0 reads GPU hidden states.
inline void layer_hidden_dump_tp_sync() {
    infinicore::context::syncStream();
    auto &ctx = global_state::get_forward_context();
    if (ctx.layer_dump_barrier != nullptr && ctx.layer_dump_tp_rank >= 0) {
        ctx.layer_dump_barrier->wait("layer_hidden_dump", ctx.layer_dump_tp_rank);
    }
}

/// Dump ``hidden_states[0, row, :]`` as float32 when ``INFINI_DUMP_LAYER_HIDDEN_DIR`` is set.
/// ``valid_seq_len`` caps the row index when the bucket is padded (piecewise prefill).
/// Optional ``segment`` writes ``{TAG}_embed_row{R}`` or ``{TAG}_layer{L}_{segment}_row{R}``.
inline void dump_layer_hidden(const infinicore::Tensor &hidden_states,
                              size_t layer_idx,
                              size_t valid_seq_len = 0,
                              const char *segment = nullptr) {
    if (!layer_hidden_dump_enabled()) {
        return;
    }
    if (infinicore::context::isGraphRecording()) {
        return;
    }

    const bool has_segment = segment != nullptr && segment[0] != '\0';
    if (has_segment) {
        const bool is_embed = std::strcmp(segment, "embed") == 0;
        if (!is_embed && !layer_hidden_dump_segment_layer_matches(layer_idx)) {
            return;
        }
    }

    const size_t seq_len = hidden_states->size(1);
    size_t row = layer_hidden_dump_row();
    if (valid_seq_len > 0 && valid_seq_len <= seq_len) {
        if (row >= valid_seq_len) {
            row = valid_seq_len - 1;
        }
    } else if (row >= seq_len) {
        return;
    }

    layer_hidden_dump_tp_sync();
    if (global_state::get_tensor_model_parallel_rank() != 0) {
        return;
    }

    auto row_tensor = hidden_states->narrow({{0, 0, 1}, {1, row, 1}});
    const auto floats = tensor_to_f32_vector(row_tensor);

    const std::string dir = layer_hidden_dump_dir();
    std::filesystem::create_directories(dir);

    std::ostringstream base;
    base << dir << "/" << layer_hidden_dump_tag();
    if (has_segment && std::strcmp(segment, "embed") == 0) {
        base << "_embed_row" << row;
    } else if (has_segment) {
        base << "_layer" << std::setfill('0') << std::setw(2) << layer_idx
             << "_" << segment << "_row" << row;
    } else {
        base << "_layer" << std::setfill('0') << std::setw(2) << layer_idx
             << "_row" << row;
    }

    const std::string bin_path = base.str() + ".f32.bin";
    std::ofstream bin_ofs(bin_path, std::ios::binary);
    if (!bin_ofs) {
        throw std::runtime_error("layer_hidden_dump: failed to open " + bin_path);
    }
    bin_ofs.write(reinterpret_cast<const char *>(floats.data()),
                  static_cast<std::streamsize>(floats.size() * sizeof(float)));

    const std::string shape_path = base.str() + ".shape.json";
    std::ofstream shape_ofs(shape_path);
    if (!shape_ofs) {
        throw std::runtime_error("layer_hidden_dump: failed to open " + shape_path);
    }
    shape_ofs << "{\"shape\": [1, 1, " << floats.size() << "], \"row\": " << row;
    if (has_segment && std::strcmp(segment, "embed") == 0) {
        shape_ofs << ", \"segment\": \"embed\"";
    } else {
        shape_ofs << ", \"layer\": " << layer_idx;
        if (has_segment) {
            shape_ofs << ", \"segment\": \"" << segment << "\"";
        }
    }
    shape_ofs << "}\n";
}

} // namespace infinilm::utils
