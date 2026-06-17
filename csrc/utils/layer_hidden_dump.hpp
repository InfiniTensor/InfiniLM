#pragma once

#include "../global_state/global_state.hpp"
#include "../global_state/parallel_state.hpp"
#include "../utils.hpp"

#include "../engine/rank_barrier.hpp"
#include "infinicore/context/context.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
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
inline bool layer_hidden_dump_all_ranks() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_DUMP_LAYER_HIDDEN_ALL_RANKS");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

/// Inclusive max TP rank to emit per-rank dump files (0 = rank0 only).
/// Unset = dump all ranks. Use ``INFINI_DUMP_LAYER_HIDDEN_MAX_RANK=1`` for ranks 0+1.
inline int layer_hidden_dump_max_rank(int tp_size) {
    static int cached = -2;
    if (cached == -2) {
        const char *raw = std::getenv("INFINI_DUMP_LAYER_HIDDEN_MAX_RANK");
        if (raw == nullptr || raw[0] == '\0') {
            cached = -1;
        } else {
            cached = static_cast<int>(std::strtol(raw, nullptr, 10));
        }
    }
    if (cached < 0) {
        return tp_size - 1;
    }
    return std::min(cached, tp_size - 1);
}

/// Whether this TP worker should write a per-rank dump file (still joins barriers).
inline bool layer_hidden_dump_this_rank_writes() {
    if (!layer_hidden_dump_all_ranks()) {
        return global_state::get_tensor_model_parallel_rank() == 0;
    }
    const int tp_size = static_cast<int>(global_state::get_tensor_model_parallel_world_size());
    const int tp_rank = static_cast<int>(global_state::get_tensor_model_parallel_rank());
    return tp_rank <= layer_hidden_dump_max_rank(tp_size);
}

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

/// Synchronize the tensor's device stream before CPU readback (TP worker threads).
inline void layer_hidden_dump_sync_tensor(const infinicore::Tensor &tensor) {
    infinicore::context::setDevice(tensor->device());
    infinicore::context::syncStream();
}

inline void layer_hidden_dump_sync_worker_device() {
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    infinicore::context::setDevice(rank_info.device);
    infinicore::context::syncStream();
}

inline std::mutex &layer_hidden_dump_d2h_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::vector<float> tensor_to_f32_vector_impl(const infinicore::Tensor &tensor) {
    layer_hidden_dump_sync_tensor(tensor);
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

inline std::vector<float> tensor_to_f32_vector(const infinicore::Tensor &tensor) {
    std::lock_guard<std::mutex> lock(layer_hidden_dump_d2h_mutex());
    return tensor_to_f32_vector_impl(tensor);
}

/// Keyed TP barrier before monolithic (EAGER) forward dump sites.
inline void eager_dump_barrier(const char *site_label) {
    layer_hidden_dump_sync_worker_device();
    auto &ctx = global_state::get_forward_context();
    if (ctx.layer_dump_barrier != nullptr && ctx.layer_dump_tp_rank >= 0) {
        ctx.layer_dump_barrier->wait(site_label, ctx.layer_dump_tp_rank);
    }
}

/// Synchronize all TP worker threads before rank-0 reads GPU hidden states.
inline void layer_hidden_dump_tp_sync(const char *site_label) {
    layer_hidden_dump_sync_worker_device();
    auto &ctx = global_state::get_forward_context();
    if (ctx.layer_dump_barrier != nullptr && ctx.layer_dump_tp_rank >= 0) {
        ctx.layer_dump_barrier->wait(site_label, ctx.layer_dump_tp_rank);
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

    const int tp_rank = global_state::get_tensor_model_parallel_rank();

    if (layer_hidden_dump_all_ranks()) {
        std::ostringstream site_key;
        site_key << "layer_hidden_dump";
        if (has_segment && std::strcmp(segment, "embed") == 0) {
            site_key << "_embed";
        } else if (has_segment) {
            site_key << "_L" << std::setfill('0') << std::setw(2) << layer_idx << "_" << segment;
        } else {
            site_key << "_L" << std::setfill('0') << std::setw(2) << layer_idx << "_full";
        }
        const std::string site = site_key.str();
        const int tp_size = static_cast<int>(global_state::get_tensor_model_parallel_world_size());
        const int max_dump_rank = layer_hidden_dump_max_rank(tp_size);
        auto &ctx = global_state::get_forward_context();
        if (ctx.layer_dump_barrier != nullptr && ctx.layer_dump_tp_rank >= 0) {
            ctx.layer_dump_barrier->wait(site.c_str(), ctx.layer_dump_tp_rank);
        }
        for (int round = 0; round <= max_dump_rank; ++round) {
            const std::string round_enter = site + "_round" + std::to_string(round);
            if (ctx.layer_dump_barrier != nullptr && ctx.layer_dump_tp_rank >= 0) {
                ctx.layer_dump_barrier->wait(round_enter.c_str(), ctx.layer_dump_tp_rank);
            }
            if (tp_rank == round) {
                std::lock_guard<std::mutex> lock(layer_hidden_dump_d2h_mutex());
                auto row_tensor = hidden_states->narrow({{0, 0, 1}, {1, row, 1}});
                layer_hidden_dump_sync_tensor(row_tensor);
                const auto floats = tensor_to_f32_vector_impl(row_tensor);

                const std::string dir = layer_hidden_dump_dir();
                std::filesystem::create_directories(dir);

                std::ostringstream base;
                base << dir << "/" << layer_hidden_dump_tag() << "_rank" << tp_rank;
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
                shape_ofs << ", \"tp_rank\": " << tp_rank;
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
            const std::string round_done = round_enter + "_done";
            if (ctx.layer_dump_barrier != nullptr && ctx.layer_dump_tp_rank >= 0) {
                ctx.layer_dump_barrier->wait(round_done.c_str(), ctx.layer_dump_tp_rank);
            }
        }
        return;
    }

    std::ostringstream site_key;
    site_key << "layer_hidden_dump";
    if (has_segment && std::strcmp(segment, "embed") == 0) {
        site_key << "_embed";
    } else if (has_segment) {
        site_key << "_L" << std::setfill('0') << std::setw(2) << layer_idx << "_" << segment;
    } else {
        site_key << "_L" << std::setfill('0') << std::setw(2) << layer_idx << "_full";
    }
    const std::string site = site_key.str();
    layer_hidden_dump_sync_tensor(hidden_states);
    layer_hidden_dump_tp_sync(site.c_str());
    if (tp_rank != 0) {
        layer_hidden_dump_tp_sync(site.c_str());
        return;
    }

    auto row_tensor = hidden_states->narrow({{0, 0, 1}, {1, row, 1}});
    layer_hidden_dump_sync_tensor(row_tensor);
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
    layer_hidden_dump_tp_sync(site.c_str());
}

/// Dump one token row from ``[1, bucket, heads, head_dim]`` staging as ``[1, 1, heads*head_dim]``.
inline void dump_staging_heads_row(const infinicore::Tensor &staging,
                                   size_t layer_idx,
                                   size_t valid_seq_len,
                                   const char *segment) {
    if (!layer_hidden_dump_enabled() || staging->shape().size() != 4) {
        return;
    }
    const size_t bucket = staging->size(1);
    const size_t heads = staging->size(2);
    const size_t head_dim = staging->size(3);
    size_t row = layer_hidden_dump_row();
    if (valid_seq_len > 0 && valid_seq_len <= bucket) {
        if (row >= valid_seq_len) {
            row = valid_seq_len - 1;
        }
    } else if (row >= bucket) {
        return;
    }
    auto row_narrow = staging->narrow({{1, row, 1}});
    auto flat = row_narrow->view({1, 1, heads * head_dim});
    // flat is [1,1,H]: pass valid_seq_len=1 so dump_layer_hidden indexes row 0.
    dump_layer_hidden(flat, layer_idx, 1, segment);
}

/// Dump one token row from ``[seq, heads, head_dim]`` (EAGER q/k after RoPE).
inline void dump_seq_heads_row(const infinicore::Tensor &seq_heads,
                               size_t layer_idx,
                               size_t valid_seq_len,
                               const char *segment) {
    if (!layer_hidden_dump_enabled() || seq_heads->shape().size() != 3) {
        return;
    }
    const size_t seq_len = seq_heads->size(0);
    const size_t heads = seq_heads->size(1);
    const size_t head_dim = seq_heads->size(2);
    size_t row = layer_hidden_dump_row();
    if (valid_seq_len > 0 && valid_seq_len <= seq_len) {
        if (row >= valid_seq_len) {
            row = valid_seq_len - 1;
        }
    } else if (row >= seq_len) {
        return;
    }
    auto row_narrow = seq_heads->narrow({{0, row, 1}});
    auto flat = row_narrow->view({1, 1, heads * head_dim});
    dump_layer_hidden(flat, layer_idx, 1, segment);
}

} // namespace infinilm::utils
