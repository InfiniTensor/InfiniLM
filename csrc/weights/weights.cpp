#include "weights.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::weights {
namespace {
Shape storage_shape(const Shape &full, Size dim, Size ranks, Size shards) {
    Shape result = full;
    if (ranks > 1 && dim < full.size()) {
        const Size parts = shards ? shards : ranks;
        if (full[dim] % parts != 0) {
            throw std::runtime_error("Weight dimension is not divisible by its checkpoint partition count");
        }
        result[dim] /= parts;
    }
    return result;
}
void loading_allowed(Runtime &runtime) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()
        || runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("Checkpoint loading is not allowed during graph recording");
    }
}
} // namespace

LoadBatch::~LoadBatch() noexcept {
    // No owner may leave this scope without proof of stream completion.
    if (count_ != 0) {
        std::terminate();
    }
}
void LoadBatch::drain() {
    if (count_ == 0) {
        return;
    }
    try {
        runtime_.syncStream();
    } catch (...) { std::terminate(); }
    for (size_t i = 0; i < count_; ++i) {
        pending_[i] = {};
    }
    count_ = 0;
}
void LoadBatch::enqueue(Tensor destination, const Tensor &source) {
    if (count_ == capacity) {
        drain();
    }
    pending_[count_] = {destination, source};
    ++count_; // Publish owners before a call that may enqueue and then throw.
    runtime_.memcpyH2D(destination->data(), source->data(), destination->nbytes(), true);
}

Weight::Weight(const Tensor &tensor, Size dim, Size rank, Size ranks, Size shards)
    : tensor_(tensor), tp_dim_(dim), tp_rank_(rank), tp_size_(ranks), num_shards_(shards) {
    if (rank >= ranks) {
        throw std::runtime_error("Weight tensor parallel rank must be less than size");
    }
}
Weight::Weight(Runtime &runtime, const Shape &shape, const DataType &dtype,
               Size dim, Size rank, Size ranks, Size shards)
    : Weight(Tensor::empty_parameter(runtime, storage_shape(shape, dim, ranks, shards), dtype),
             dim, rank, ranks, shards) {}

Tensor Weight::select_source(Runtime &runtime, const Tensor &source) const {
    if (tensor_.empty() || source.empty()) {
        throw std::invalid_argument("Cannot load an empty weight handle");
    }
    if (runtime.device() != tensor_->device()) {
        throw std::invalid_argument("Weight load Runtime and destination devices differ");
    }
    if (tensor_->dtype() != source->dtype()) {
        throw std::runtime_error("Dtype mismatch loading weight: " + tensor_->info() + ", source: " + source->info());
    }
    Shape expected = tensor_->shape();
    const bool replicated = tp_dim_ < expected.size() && num_shards_ > 0 && num_shards_ < tp_size_;
    if (tp_dim_ < expected.size()) {
        expected[tp_dim_] *= replicated ? num_shards_ : tp_size_;
    }
    Tensor selected = source;
    if (expected != source->shape()) {
        const auto &actual = source->shape();
        const bool singleton_transpose = !replicated && expected.size() == 2 && actual.size() == 2
                                      && expected[0] == actual[1] && expected[1] == actual[0]
                                      && (expected[0] == 1 || expected[1] == 1);
        if (!singleton_transpose) {
            throw std::runtime_error("Shape mismatch loading weight: " + tensor_->info() + ", source: " + source->info());
        }
        if (source->is_contiguous()) {
            selected = source->view(expected);
        } else if (source->device() == runtime.device()) {
            selected = source->contiguous(runtime)->view(expected);
        } else {
            selected = source->permute({1, 0}); // Preserve external CPU owner and strides.
        }
    }
    if (tp_size_ > 1 && tp_dim_ < expected.size()) {
        const Size shard = replicated ? tp_rank_ / (tp_size_ / num_shards_) : tp_rank_;
        const Size length = tensor_->size(tp_dim_);
        const Size offset = shard * length;
        if (offset > selected->size(tp_dim_) || length > selected->size(tp_dim_) - offset) {
            throw std::runtime_error("Weight tensor parallel slice is out of bounds");
        }
        selected = selected->narrow({{tp_dim_, offset, length}});
    }
    return selected;
}
void Weight::load_into(Runtime &runtime, const Tensor &source, LoadBatch *batch) {
    runtime.assertOwnerThread();
    const Tensor selected = select_source(runtime, source);
    const bool asynchronous = batch && tp_dim_ != 1
                           && runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                           && selected->device().type() == infinilm::runtime::Device::Type::kCpu
                           && selected->dtype() == tensor_->dtype() && selected->shape() == tensor_->shape()
                           && selected->is_contiguous() && tensor_->is_contiguous();
    if (asynchronous) {
        if (tensor_->nbytes()) {
            batch->enqueue(tensor_, selected);
        }
    } else {
        if (batch) {
            batch->drain();
        }
        tensor_->copy_from(runtime, selected);
    }
}
void Weight::load_no_sync(Runtime &runtime, const Tensor &source) { load_into(runtime, source, nullptr); }
void Weight::load(Runtime &runtime, const Tensor &source) {
    load_no_sync(runtime, source);
    runtime.syncStream();
}

Tensor ModelWeights::bind_weight(const std::string &name, Weight weight) {
    local_weights_[name] = weight;
    return weight;
}
void ModelWeights::collect(std::unordered_map<std::string, Weight> &out, const std::string &prefix) const {
    for (const auto &[name, weight] : local_weights_) {
        out[prefix + name] = weight;
    }
    for (const auto &[name, child] : child_weights_) {
        child->collect(out, prefix + name + ".");
    }
}
std::unordered_map<std::string, Weight> ModelWeights::state_dict() const {
    std::unordered_map<std::string, Weight> result;
    collect(result, "");
    return result;
}
std::vector<std::string> ModelWeights::state_dict_keys() const {
    std::vector<std::string> result;
    for (const auto &[name, weight] : state_dict()) {
        result.push_back(name);
    }
    return result;
}
std::vector<std::string> ModelWeights::state_required_keys() const {
    std::vector<std::string> result;
    for (const auto &[name, weight] : state_dict()) {
        if (weight.checkpoint_required()) {
            result.push_back(name);
        }
    }
    return result;
}
Weight &ModelWeights::resolve(const std::string &name) {
    auto local = local_weights_.find(name);
    if (local != local_weights_.end()) {
        return local->second;
    }
    std::shared_ptr<ModelWeights> selected;
    size_t prefix = 0;
    for (const auto &[key, child] : child_weights_) {
        if (key.size() > prefix && name.size() > key.size()
            && name.compare(0, key.size(), key) == 0 && name[key.size()] == '.') {
            selected = child;
            prefix = key.size();
        }
    }
    if (selected) {
        return selected->resolve(name.substr(prefix + 1));
    }
    throw std::runtime_error("Weight '" + name + "' not found in model");
}
void ModelWeights::load_parameter(Runtime &runtime, const std::string &name, const Tensor &source) {
    runtime.assertOwnerThread();
    try {
        resolve(name).load(runtime, source);
    } catch (const std::exception &e) { throw std::runtime_error("Error loading parameter '" + name + "': " + e.what()); }
}
void ModelWeights::load_parameters_no_sync(Runtime &runtime,
                                           const std::unordered_map<std::string, Tensor> &sources, bool strict) {
    loading_allowed(runtime);
    auto bindings = state_dict();
    LoadBatch batch(runtime);
    try {
        for (const auto &[name, source] : sources) {
            auto binding = bindings.find(name);
            if (binding == bindings.end()) {
                if (strict) {
                    throw std::runtime_error("Weight '" + name + "' not found in model");
                }
                continue;
            }
            try {
                binding->second.load_into(runtime, source, &batch);
            } catch (const std::exception &e) { throw std::runtime_error("Error loading parameter '" + name + "': " + e.what()); }
        }
        batch.drain();
    } catch (...) {
        batch.drain();
        throw;
    }
}
} // namespace infinilm::weights
