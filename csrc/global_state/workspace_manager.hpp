#pragma once

#include "../models/infinilm_model.hpp"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace infinilm::global_state {

/**
 * @brief Unified GPU inference scratch buffer.
 *
 * Flow: register_buffer -> finalize_and_bind -> get_buffer (named cache) -> log_registrations.
 * get_buffer looks up buffer_name; on miss bump-allocates and caches for reuse across layers.
 */
class WorkspaceManager {
public:
    WorkspaceManager() = default;
    ~WorkspaceManager() = default;

    /** @brief Register a bump slot at current total_bytes_. Same name reuses one slot. */
    void register_buffer(const std::string &name,
                         const infinicore::Shape &shape,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device);

    /** @brief Register a pinned@0 slot (only offset==0). May overlap bump slots. */
    void register_buffer(const std::string &name,
                         size_t offset,
                         const infinicore::Shape &shape,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device);

    /** @brief Return a cached view by name, or bump-allocate and cache on first use. */
    infinicore::Tensor get_buffer(const std::string &buffer_name,
                                  const infinicore::Shape &shape,
                                  const infinicore::DataType &dtype,
                                  const infinicore::Device &device);

    /** @brief Return a cached view by name, or bind at offset and cache on first use. */
    infinicore::Tensor get_buffer(const std::string &buffer_name,
                                  size_t offset,
                                  const infinicore::Shape &shape,
                                  const infinicore::DataType &dtype,
                                  const infinicore::Device &device);

    /** @brief Allocate scratch_buffer_. */
    void finalize_and_bind();

    /** @brief Reset runtime bump offset to 0. Call at the start of each forward. */
    void reset_runtime_buffers();

    /** @brief Log slot layout with memory ranges and overlap info. */
    void log_registrations() const;

private:
    struct BufferRegistration {
        size_t offset{0};        // view start in scratch_buffer_
        size_t aligned_bytes{0}; // aligned byte span
        bool is_bump_tail{true}; // bump vs pinned@0
        infinicore::Shape shape;
        infinicore::DataType dtype;
        infinicore::Device device;
        infinicore::Tensor bound_view; // set in finalize_and_bind
    };

    void _register_buffer_impl(const std::string &name,
                               size_t offset,
                               const infinicore::Shape &shape,
                               const infinicore::DataType &dtype,
                               const infinicore::Device &device,
                               bool bump_tail);

    infinicore::Tensor _make_runtime_view(size_t offset,
                                          const infinicore::Shape &shape,
                                          const infinicore::DataType &dtype,
                                          const infinicore::Device &device);

    bool finalized_{false};

    infinicore::Tensor scratch_buffer_;
    size_t total_bytes_{0};
    size_t scratch_buffer_offset_{0};

    std::unordered_map<std::string, BufferRegistration> registrations_;
    std::unordered_map<std::string, infinicore::Tensor> runtime_buffers_;
};

} // namespace infinilm::global_state
