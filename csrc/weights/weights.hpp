#pragma once

#include <array>
#include <memory>
#include <runtime/tensor.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infinilm::weights {
using infinilm::runtime::DataType;
using infinilm::runtime::Runtime;
using infinilm::runtime::Shape;
using infinilm::runtime::Size;
using infinilm::runtime::Tensor;

// One loading transaction retains only outstanding asynchronous transfers.
class LoadBatch final {
public:
    static constexpr size_t capacity = 256;
    explicit LoadBatch(Runtime &runtime) : runtime_(runtime) {}
    ~LoadBatch() noexcept;
    LoadBatch(const LoadBatch &) = delete;
    LoadBatch &operator=(const LoadBatch &) = delete;
    void drain();

private:
    friend class Weight;
    struct Owners {
        Tensor destination;
        Tensor source;
    };
    Runtime &runtime_;
    std::array<Owners, capacity> pending_{};
    size_t count_ = 0;
    void enqueue(Tensor destination, const Tensor &source);
};

// LM checkpoint binding: physical tensor storage plus checkpoint shard policy.
// Composition keeps tensor operations in Core and loading policy in the model.
class Weight final {
public:
    Weight() = default;
    bool checkpoint_required() const noexcept { return checkpoint_required_; }
    Weight(const Tensor &tensor, Size tp_dim = 0, Size tp_rank = 0,
           Size tp_size = 1, Size num_shards = 0);
    Weight(Runtime &runtime, const Shape &shape, const DataType &dtype,
           Size tp_dim = 0, Size tp_rank = 0, Size tp_size = 1, Size num_shards = 0);
    operator Tensor() const { return tensor_; }
    const Tensor &tensor() const noexcept { return tensor_; }
    Tensor &tensor() noexcept { return tensor_; }
    operator bool() const { return static_cast<bool>(tensor_); }
    infinilm::runtime::TensorImpl *operator->() { return tensor_.operator->(); }
    const infinilm::runtime::TensorImpl *operator->() const { return tensor_.operator->(); }
    bool empty() const noexcept { return tensor_.empty(); }
    size_t use_count() const noexcept { return tensor_.use_count(); }
    void reset() noexcept { tensor_.reset(); }
    void reset(std::shared_ptr<infinilm::runtime::TensorImpl> tensor) noexcept { tensor_.reset(std::move(tensor)); }
    void load(Runtime &runtime, const Tensor &source);
    void load_no_sync(Runtime &runtime, const Tensor &source);

private:
    friend class ModelWeights;
    bool checkpoint_required_ = true;
    Tensor tensor_;
    Size tp_dim_ = 0, tp_rank_ = 0, tp_size_ = 1, num_shards_ = 0;
    Tensor select_source(Runtime &runtime, const Tensor &source) const;
    void load_into(Runtime &runtime, const Tensor &source, LoadBatch *batch);
};

// The named weight tree follows LM ownership, and supports actual model RTTI.
// It has no execution, device, buffer, or layer-construction responsibilities.
class ModelWeights {
public:
    virtual ~ModelWeights() = default;
    std::unordered_map<std::string, Weight> state_dict() const;
    std::vector<std::string> state_dict_keys() const;
    std::vector<std::string> state_required_keys() const;
    void load_parameter(Runtime &runtime, const std::string &name, const Tensor &source);
    void load_parameters_no_sync(Runtime &runtime,
                                 const std::unordered_map<std::string, Tensor> &sources,
                                 bool strict = true);
    const std::unordered_map<std::string, std::shared_ptr<ModelWeights>> &children() const {
        return child_weights_;
    }
    // Loading owners include fused projections that do not add checkpoint prefixes.
    virtual std::vector<std::shared_ptr<ModelWeights>> loading_children() const {
        std::vector<std::shared_ptr<ModelWeights>> result;
        for (const auto &[_, child] : child_weights_) {
            result.push_back(child);
        }
        return result;
    }
    // Model traversal visits bound children first; hooks do not revisit them.
    virtual void process_weights_after_loading(Runtime &, bool) {}

protected:
    Tensor bind_weight(const std::string &name, Weight weight);
    Tensor bind_derived_weight(const std::string &name, Weight weight) {
        weight.checkpoint_required_ = false;
        return bind_weight(name, std::move(weight));
    }
    template <class T>
    std::shared_ptr<T> bind_child(const std::string &name, std::shared_ptr<T> child) {
        static_assert(std::is_base_of<ModelWeights, T>::value,
                      "LM weight children must derive from ModelWeights");
        if (!child) {
            throw std::invalid_argument("Cannot bind a null weight child");
        }
        child_weights_[name] = child;
        return child;
    }
    std::unordered_map<std::string, Weight> local_weights_;
    std::unordered_map<std::string, std::shared_ptr<ModelWeights>> child_weights_;

private:
    void collect(std::unordered_map<std::string, Weight> &out, const std::string &prefix) const;
    Weight &resolve(const std::string &name);
};
} // namespace infinilm::weights
