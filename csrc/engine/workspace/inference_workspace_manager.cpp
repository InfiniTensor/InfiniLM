#include "inference_workspace_manager.hpp"

#include "../../utils.hpp"

#include "infinicore/context/context.hpp"

#include <stdexcept>
#include <utility>

namespace infinilm::engine {

namespace {

size_t numel_of(const infinicore::Shape &shape) {
    size_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    return numel;
}

class ZeroTensorGraphOperator final : public infinicore::graph::GraphOperator {
public:
    explicit ZeroTensorGraphOperator(const infinicore::Tensor &tensor)
        : tensor_(tensor), bytes_(tensor->nbytes()) {}

    void run() const override {
        infinicore::context::setDeviceMemoryAsync(
            const_cast<std::byte *>(tensor_->data()), 0, bytes_, infinicore::context::getStream());
    }

private:
    infinicore::graph::GraphTensor tensor_;
    size_t bytes_ = 0;
};

} // namespace

infinicore::Tensor WorkspaceBuffer::as_tensor(const infinicore::Shape &shape, infinicore::DataType dtype) const {
    const auto requested_bytes = numel_of(shape) * infinicore::dsize(dtype);
    if (requested_bytes > bytes) {
        throw std::runtime_error("workspace buffer is too small for requested tensor view");
    }
    if (owner && dtype == infinicore::DataType::U8 && shape.size() == 1 && shape[0] == requested_bytes) {
        return owner->narrow({{0, 0, requested_bytes}});
    }
    return infinicore::Tensor::from_blob(ptr, shape, dtype, device);
}

InferenceWorkspaceManager::InferenceWorkspaceManager(infinicore::Device device) : device_(device) {}

void InferenceWorkspaceManager::begin_forward() {
    if (in_forward_) {
        throw std::runtime_error("InferenceWorkspaceManager::begin_forward called while a forward is active");
    }
    reclaim_retired_typed_blocks();
    in_forward_ = true;
    transient_cursor_ = 0;
    collective_cursor_ = 0;
}

void InferenceWorkspaceManager::end_forward() {
    for (auto it = transient_active_.rbegin(); it != transient_active_.rend(); ++it) {
        auto &block = *it;
        block.active = false;
        if (block.typed) {
            if (block.tensor.use_count() == 1) {
                typed_transient_free_[block.key].push_back(std::move(block));
            } else {
                transient_retired_.push_back(std::move(block));
            }
        } else {
            transient_free_[block.capacity].push_back(std::move(block));
        }
    }
    transient_active_.clear();
    in_forward_ = false;
}

void InferenceWorkspaceManager::begin_warmup() {
    if (in_forward_) {
        throw std::runtime_error("cannot begin workspace warmup while a forward is active");
    }
    transient_plan_.clear();
    transient_cursor_ = 0;
    mode_ = WorkspaceMode::Warmup;
}

void InferenceWorkspaceManager::end_warmup() {
    if (in_forward_) {
        throw std::runtime_error("cannot end workspace warmup while a forward is active");
    }
    mode_ = WorkspaceMode::Dynamic;
    transient_cursor_ = 0;
}

void InferenceWorkspaceManager::freeze_transient_pool() {
    if (in_forward_) {
        throw std::runtime_error("cannot freeze workspace pool while a forward is active");
    }
    mode_ = WorkspaceMode::Frozen;
    transient_cursor_ = 0;
}

void InferenceWorkspaceManager::unfreeze_transient_pool() {
    mode_ = WorkspaceMode::Dynamic;
    transient_cursor_ = 0;
}

WorkspaceBuffer InferenceWorkspaceManager::acquire_temp_buffer(
    std::string_view tag,
    size_t bytes,
    size_t alignment,
    WorkspaceZeroPolicy zero_policy) {
    if (!in_forward_) {
        throw std::runtime_error("transient inference workspace can only be acquired during forward: tag=" + std::string(tag) + ", scope=" + collective_scope_ + ", graph_recording=" + (infinicore::context::isGraphRecording() ? std::string("true") : std::string("false")));
    }
    const auto capacity = normalize_bytes(bytes, alignment);
    auto &free_list = transient_free_[capacity];

    if (mode_ == WorkspaceMode::Warmup) {
        transient_plan_.push_back(TransientRequest{std::string(tag), bytes, capacity});
    } else if (mode_ == WorkspaceMode::Frozen) {
        if (transient_cursor_ >= transient_plan_.size()) {
            throw std::runtime_error("frozen workspace pool received more transient requests than warmup recorded");
        }
        const auto &expected = transient_plan_[transient_cursor_];
        if (expected.capacity != capacity) {
            throw std::runtime_error(
                "frozen workspace pool request size mismatch for " + std::string(tag) +
                ": expected capacity " + std::to_string(expected.capacity) +
                ", got " + std::to_string(capacity));
        }
    }
    ++transient_cursor_;

    Block block;
    if (!free_list.empty()) {
        block = std::move(free_list.back());
        free_list.pop_back();
    } else {
        if (mode_ == WorkspaceMode::Frozen) {
            throw std::runtime_error("frozen workspace pool has no free block for " + std::string(tag));
        }
        block.tensor = infinicore::Tensor::empty({capacity}, infinicore::DataType::U8, device_);
        block.capacity = capacity;
        block.lifetime = WorkspaceLifetime::Transient;
    }

    block.owner = "transient";
    block.key = std::string(tag);
    block.active = true;
    if (zero_policy == WorkspaceZeroPolicy::OnAcquire || zero_policy == WorkspaceZeroPolicy::OnCreate) {
        zero_tensor(block.tensor);
    }

    transient_active_.push_back(std::move(block));
    return make_buffer(transient_active_.back(), bytes);
}

infinicore::Tensor InferenceWorkspaceManager::acquire_temp_tensor(
    std::string_view tag,
    const infinicore::Shape &shape,
    infinicore::DataType dtype,
    size_t /*alignment*/,
    WorkspaceZeroPolicy zero_policy) {
    if (!in_forward_) {
        throw std::runtime_error("transient inference tensor can only be acquired during forward: tag=" + std::string(tag) + ", scope=" + collective_scope_ + ", graph_recording=" + (infinicore::context::isGraphRecording() ? std::string("true") : std::string("false")));
    }

    const auto bytes = numel_of(shape) * infinicore::dsize(dtype);
    const auto key = typed_key(shape, dtype);
    auto &free_list = typed_transient_free_[key];

    if (mode_ == WorkspaceMode::Warmup) {
        transient_plan_.push_back(TransientRequest{std::string(tag), bytes, bytes});
    } else if (mode_ == WorkspaceMode::Frozen) {
        if (transient_cursor_ >= transient_plan_.size()) {
            throw std::runtime_error("frozen workspace pool received more transient tensor requests than warmup recorded");
        }
        const auto &expected = transient_plan_[transient_cursor_];
        if (expected.capacity != bytes) {
            throw std::runtime_error(
                "frozen workspace pool tensor size mismatch for " + std::string(tag) +
                ": expected " + std::to_string(expected.capacity) +
                ", got " + std::to_string(bytes));
        }
    }
    ++transient_cursor_;

    Block block;
    if (!free_list.empty()) {
        block = std::move(free_list.back());
        free_list.pop_back();
    } else {
        if (mode_ == WorkspaceMode::Frozen) {
            throw std::runtime_error("frozen workspace pool has no free typed tensor for " + std::string(tag));
        }
        block.tensor = infinicore::Tensor::empty(shape, dtype, device_);
        block.capacity = bytes;
        block.lifetime = WorkspaceLifetime::Transient;
        block.typed = true;
    }

    block.owner = "transient";
    block.key = key;
    block.active = true;
    if (zero_policy == WorkspaceZeroPolicy::OnAcquire || zero_policy == WorkspaceZeroPolicy::OnCreate) {
        zero_tensor(block.tensor);
    }

    transient_active_.push_back(std::move(block));
    return transient_active_.back().tensor;
}

void InferenceWorkspaceManager::set_collective_scope(std::string_view scope) {
    collective_scope_ = std::string(scope);
}

void InferenceWorkspaceManager::clear_collective_scope() {
    collective_scope_.clear();
}

std::string InferenceWorkspaceManager::next_collective_key(std::string_view tag) {
    if (!in_forward_) {
        throw std::runtime_error("collective workspace keys can only be acquired during forward");
    }
    std::string key;
    if (!collective_scope_.empty()) {
        key += collective_scope_;
        key += ":";
    }
    key += std::string(tag);
    key += ".";
    key += std::to_string(collective_cursor_++);
    return key;
}

WorkspaceBuffer InferenceWorkspaceManager::get_persistent_buffer(
    std::string_view owner,
    std::string_view key,
    size_t bytes,
    WorkspaceLifetime lifetime,
    WorkspaceZeroPolicy zero_policy) {
    if (lifetime == WorkspaceLifetime::Transient) {
        throw std::runtime_error("persistent workspace request cannot use transient lifetime");
    }

    const auto map_key = full_key(owner, key);
    const auto capacity = normalize_bytes(bytes, 256);
    auto it = persistent_.find(map_key);
    if (it == persistent_.end()) {
        Block block;
        block.tensor = infinicore::Tensor::empty({capacity}, infinicore::DataType::U8, device_);
        block.capacity = capacity;
        block.owner = std::string(owner);
        block.key = map_key;
        block.lifetime = lifetime;
        if (zero_policy == WorkspaceZeroPolicy::OnCreate || zero_policy == WorkspaceZeroPolicy::OnAcquire) {
            zero_tensor(block.tensor);
        }
        it = persistent_.emplace(map_key, std::move(block)).first;
    } else {
        auto &block = it->second;
        if (block.owner != std::string(owner)) {
            throw std::runtime_error("persistent workspace owner mismatch for key " + map_key);
        }
        if (block.lifetime != lifetime) {
            throw std::runtime_error("persistent workspace lifetime mismatch for key " + map_key);
        }
        if (block.capacity < capacity) {
            if (graph_frozen_) {
                throw std::runtime_error("cannot grow workspace after graph freeze: " + map_key);
            }
            block.tensor = infinicore::Tensor::empty({capacity}, infinicore::DataType::U8, device_);
            block.capacity = capacity;
            if (zero_policy == WorkspaceZeroPolicy::OnCreate || zero_policy == WorkspaceZeroPolicy::OnAcquire) {
                zero_tensor(block.tensor);
            }
        } else if (zero_policy == WorkspaceZeroPolicy::OnAcquire) {
            zero_tensor(block.tensor);
        }
    }
    return make_buffer(it->second, bytes);
}

infinicore::Tensor InferenceWorkspaceManager::get_persistent_tensor(
    std::string_view owner,
    std::string_view key,
    const infinicore::Shape &shape,
    infinicore::DataType dtype,
    WorkspaceLifetime lifetime,
    WorkspaceZeroPolicy zero_policy) {
    const auto bytes = numel_of(shape) * infinicore::dsize(dtype);
    if (dtype == infinicore::DataType::U8 && shape.size() == 1) {
        if (lifetime == WorkspaceLifetime::Transient) {
            throw std::runtime_error("persistent workspace request cannot use transient lifetime");
        }
        const auto map_key = full_key(owner, key);
        auto it = persistent_.find(map_key);
        if (it == persistent_.end()) {
            Block block;
            block.tensor = infinicore::Tensor::empty(shape, dtype, device_);
            block.capacity = bytes;
            block.owner = std::string(owner);
            block.key = map_key;
            block.lifetime = lifetime;
            if (zero_policy == WorkspaceZeroPolicy::OnCreate || zero_policy == WorkspaceZeroPolicy::OnAcquire) {
                zero_tensor(block.tensor);
            }
            it = persistent_.emplace(map_key, std::move(block)).first;
        } else {
            auto &block = it->second;
            if (block.owner != std::string(owner)) {
                throw std::runtime_error("persistent workspace owner mismatch for key " + map_key);
            }
            if (block.lifetime != lifetime) {
                throw std::runtime_error("persistent workspace lifetime mismatch for key " + map_key);
            }
            if (block.capacity < bytes) {
                if (graph_frozen_) {
                    throw std::runtime_error("cannot grow workspace after graph freeze: " + map_key);
                }
                block.tensor = infinicore::Tensor::empty(shape, dtype, device_);
                block.capacity = bytes;
                if (zero_policy == WorkspaceZeroPolicy::OnCreate || zero_policy == WorkspaceZeroPolicy::OnAcquire) {
                    zero_tensor(block.tensor);
                }
            } else if (zero_policy == WorkspaceZeroPolicy::OnAcquire) {
                zero_tensor(block.tensor);
            }
        }
        return it->second.tensor;
    }

    return get_persistent_buffer(owner, key, bytes, lifetime, zero_policy).as_tensor(shape, dtype);
}

void InferenceWorkspaceManager::reserve_persistent_buffer(
    std::string_view owner,
    std::string_view key,
    size_t bytes,
    WorkspaceLifetime lifetime,
    WorkspaceZeroPolicy zero_policy) {
    (void)get_persistent_buffer(owner, key, bytes, lifetime, zero_policy);
}

void InferenceWorkspaceManager::freeze_for_graph() {
    graph_frozen_ = true;
}

void InferenceWorkspaceManager::unfreeze_graph() {
    graph_frozen_ = false;
}

void InferenceWorkspaceManager::zero(std::string_view key) {
    auto it = persistent_.find(std::string(key));
    if (it != persistent_.end()) {
        zero_tensor(it->second.tensor);
    }
}

void InferenceWorkspaceManager::zero_by_prefix(std::string_view prefix) {
    for (auto &[key, block] : persistent_) {
        if (key.compare(0, prefix.size(), prefix.data(), prefix.size()) == 0) {
            zero_tensor(block.tensor);
        }
    }
}

WorkspaceStats InferenceWorkspaceManager::stats() const {
    WorkspaceStats stats;
    for (const auto &[capacity, blocks] : transient_free_) {
        stats.transient_reserved_bytes += capacity * blocks.size();
        stats.transient_cached_blocks += blocks.size();
    }
    for (const auto &[_, blocks] : typed_transient_free_) {
        for (const auto &block : blocks) {
            stats.transient_reserved_bytes += block.capacity;
            ++stats.transient_cached_blocks;
        }
    }
    for (const auto &block : transient_retired_) {
        stats.transient_reserved_bytes += block.capacity;
    }
    for (const auto &block : transient_active_) {
        stats.transient_reserved_bytes += block.capacity;
        stats.transient_active_bytes += block.capacity;
        ++stats.transient_active_blocks;
    }
    for (const auto &[_, block] : persistent_) {
        stats.persistent_reserved_bytes += block.capacity;
        ++stats.persistent_blocks;
    }
    return stats;
}

size_t InferenceWorkspaceManager::normalize_bytes(size_t bytes, size_t alignment) const {
    if (bytes == 0) {
        return 0;
    }
    if (alignment == 0) {
        alignment = 1;
    }
    return ((bytes + alignment - 1) / alignment) * alignment;
}

WorkspaceBuffer InferenceWorkspaceManager::make_buffer(Block &block, size_t requested_bytes) {
    return WorkspaceBuffer{block.tensor->data(), requested_bytes, device_, block.tensor};
}

void InferenceWorkspaceManager::zero_tensor(infinicore::Tensor &tensor) const {
    if (infinicore::context::isGraphRecording()) {
        infinicore::context::addGraphOperator(std::make_shared<ZeroTensorGraphOperator>(tensor));
    } else {
        set_zeros_device_async(tensor);
    }
}

std::string InferenceWorkspaceManager::typed_key(const infinicore::Shape &shape, infinicore::DataType dtype) const {
    std::string key = std::to_string(static_cast<int>(dtype));
    for (auto dim : shape) {
        key += ":" + std::to_string(dim);
    }
    return key;
}

void InferenceWorkspaceManager::reclaim_retired_typed_blocks() {
    std::vector<Block> still_retired;
    for (auto &block : transient_retired_) {
        if (block.tensor.use_count() == 1) {
            typed_transient_free_[block.key].push_back(std::move(block));
        } else {
            still_retired.push_back(std::move(block));
        }
    }
    transient_retired_ = std::move(still_retired);
}

std::string InferenceWorkspaceManager::full_key(std::string_view owner, std::string_view key) const {
    return std::string(owner) + "." + std::string(key);
}

} // namespace infinilm::engine
