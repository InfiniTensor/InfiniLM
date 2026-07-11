#pragma once

#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infinilm::engine {

enum class WorkspaceLifetime {
    Transient,
    Persistent,
    GraphPersistent,
};

enum class WorkspaceZeroPolicy {
    None,
    OnCreate,
    OnAcquire,
};

enum class WorkspaceMode {
    Dynamic,
    Warmup,
    Frozen,
};

struct WorkspaceStats {
    size_t transient_reserved_bytes = 0;
    size_t persistent_reserved_bytes = 0;
    size_t transient_active_bytes = 0;
    size_t transient_active_blocks = 0;
    size_t transient_cached_blocks = 0;
    size_t persistent_blocks = 0;
};

struct WorkspaceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    infinicore::Device device;
    infinicore::Tensor owner;

    template <typename T>
    T *as() const {
        return reinterpret_cast<T *>(ptr);
    }

    infinicore::Tensor as_tensor(const infinicore::Shape &shape, infinicore::DataType dtype) const;
};

class InferenceWorkspaceManager {
public:
    explicit InferenceWorkspaceManager(infinicore::Device device);

    void begin_forward();
    void end_forward();

    void begin_warmup();
    void end_warmup();
    void freeze_transient_pool();
    void unfreeze_transient_pool();

    WorkspaceBuffer acquire_temp_buffer(
        std::string_view tag,
        size_t bytes,
        size_t alignment = 256,
        WorkspaceZeroPolicy zero_policy = WorkspaceZeroPolicy::None);

    infinicore::Tensor acquire_temp_tensor(
        std::string_view tag,
        const infinicore::Shape &shape,
        infinicore::DataType dtype,
        size_t alignment = 256,
        WorkspaceZeroPolicy zero_policy = WorkspaceZeroPolicy::None);

    void set_collective_scope(std::string_view scope);
    void clear_collective_scope();
    std::string next_collective_key(std::string_view tag);

    WorkspaceBuffer get_persistent_buffer(
        std::string_view owner,
        std::string_view key,
        size_t bytes,
        WorkspaceLifetime lifetime = WorkspaceLifetime::Persistent,
        WorkspaceZeroPolicy zero_policy = WorkspaceZeroPolicy::None);

    infinicore::Tensor get_persistent_tensor(
        std::string_view owner,
        std::string_view key,
        const infinicore::Shape &shape,
        infinicore::DataType dtype,
        WorkspaceLifetime lifetime = WorkspaceLifetime::Persistent,
        WorkspaceZeroPolicy zero_policy = WorkspaceZeroPolicy::None);

    void reserve_persistent_buffer(
        std::string_view owner,
        std::string_view key,
        size_t bytes,
        WorkspaceLifetime lifetime = WorkspaceLifetime::Persistent,
        WorkspaceZeroPolicy zero_policy = WorkspaceZeroPolicy::None);

    void freeze_for_graph();
    void unfreeze_graph();

    void zero(std::string_view key);
    void zero_by_prefix(std::string_view prefix);

    WorkspaceStats stats() const;

private:
    struct Block {
        infinicore::Tensor tensor;
        size_t capacity = 0;
        std::string owner;
        std::string key;
        WorkspaceLifetime lifetime = WorkspaceLifetime::Transient;
        bool active = false;
        bool typed = false;
    };

    struct TransientRequest {
        std::string tag;
        size_t bytes = 0;
        size_t capacity = 0;
    };

    size_t normalize_bytes(size_t bytes, size_t alignment) const;
    WorkspaceBuffer make_buffer(Block &block, size_t requested_bytes);
    std::string typed_key(const infinicore::Shape &shape, infinicore::DataType dtype) const;
    void reclaim_retired_typed_blocks();
    void zero_tensor(infinicore::Tensor &tensor) const;
    std::string full_key(std::string_view owner, std::string_view key) const;

    infinicore::Device device_;
    bool in_forward_ = false;
    bool graph_frozen_ = false;
    WorkspaceMode mode_ = WorkspaceMode::Dynamic;
    size_t transient_cursor_ = 0;
    size_t collective_cursor_ = 0;
    std::string collective_scope_;

    std::unordered_map<size_t, std::vector<Block>> transient_free_;
    std::unordered_map<std::string, std::vector<Block>> typed_transient_free_;
    std::vector<Block> transient_active_;
    std::vector<Block> transient_retired_;
    std::unordered_map<std::string, Block> persistent_;
    std::vector<TransientRequest> transient_plan_;
};

} // namespace infinilm::engine
