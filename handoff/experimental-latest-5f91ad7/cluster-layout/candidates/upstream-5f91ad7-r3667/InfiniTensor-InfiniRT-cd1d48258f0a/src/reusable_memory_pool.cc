#include <infini/rt/device_guard.h>
#include <infini/rt/reusable_memory_pool.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace infini::rt {
namespace {
void check(runtime::Error status, const char* operation) {
  if (status != runtime::kSuccess)
    throw std::runtime_error(std::string(operation) +
                             " failed with error code " +
                             std::to_string(static_cast<long long>(status)));
}
void warn(runtime::Error status, const char* operation) noexcept {
  if (status != runtime::kSuccess)
    std::fprintf(stderr, "ReusableMemoryPool cleanup %s error=%lld\n",
                 operation, static_cast<long long>(status));
}
std::size_t align_up(std::size_t size, std::size_t alignment) {
  if (size > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    throw std::length_error("allocator request exceeds addressable size");
  return (size + alignment - 1) / alignment * alignment;
}
}  // namespace

struct ReusableMemoryPool::State {
  struct Block {
    enum class Kind { SizeClass, Large, Exact };
    void* ptr = nullptr;
    std::size_t size = 0;
    Kind kind = Kind::Large;
    std::size_t class_index = 0;
    std::size_t pin_count = 0;
    bool in_use = false;
    std::size_t use_count = 0;
    uint64_t allocation_capture_generation = 0;
  };
  struct SizeClass {
    std::size_t block_size;
    std::vector<std::shared_ptr<Block>> free_blocks;
  };
  Device device_;
  runtime::Stream stream_ = nullptr;
  runtime::Stream allocation_stream_ = nullptr;
  bool pinned_mode_ = false;
  std::thread::id pin_owner_;
  uint64_t capture_generation_ = 0;
  std::vector<SizeClass> size_classes_;
  std::vector<std::shared_ptr<Block>> large_blocks_;
  std::unordered_map<void*, std::shared_ptr<Block>> all_blocks_;
  std::vector<std::shared_ptr<Block>> capture_frozen_blocks_;
  std::unordered_set<Block*> capture_frozen_block_set_;
  std::mutex mutex_;

  explicit State(Device device) : device_(device) {
    size_classes_ = {
        {32 * 1024, {}},         {256 * 1024, {}},       {1 * 1024 * 1024, {}},
        {2 * 1024 * 1024, {}},   {4 * 1024 * 1024, {}},  {8 * 1024 * 1024, {}},
        {16 * 1024 * 1024, {}},  {32 * 1024 * 1024, {}}, {64 * 1024 * 1024, {}},
        {128 * 1024 * 1024, {}}, {256 * 1024 * 1024, {}}};
    if (device_.type() == Device::Type::kCount)
      throw std::invalid_argument("invalid pool device type");
    DeviceGuard guard{device_.type(), device_.index(), true};
    const auto status = runtime::StreamCreate(&stream_);
    if (status != runtime::kSuccess && stream_ != nullptr) {
      warn(runtime::StreamDestroy(stream_), "partial stream destroy");
      stream_ = nullptr;
    }
    check(status, "create pool stream");
    allocation_stream_ =
        device_.type() == Device::Type::kNvidia ? stream_ : nullptr;
  }

  runtime::Error allocate_block_(void** ptr, std::size_t size) {
#if defined(WITH_NVIDIA)
    if (allocation_stream_ != nullptr) {
      cudaStreamCaptureStatus status;
      check(static_cast<runtime::Error>(cudaStreamIsCapturing(
                reinterpret_cast<cudaStream_t>(allocation_stream_), &status)),
            "query pool stream capture");
      // A captured cudaMallocAsync belongs to the graph, not this pool.
      if (status != cudaStreamCaptureStatusNone)
        throw std::runtime_error(
            "pool cannot grow during CUDA stream capture; preallocate blocks "
            "before capture");
    }
#endif
    return allocation_stream_ != nullptr
               ? runtime::MallocAsync(ptr, size, allocation_stream_)
               : runtime::Malloc(ptr, size);
  }
  runtime::Error free_block_(void* ptr) {
    return allocation_stream_ != nullptr
               ? runtime::FreeAsync(ptr, allocation_stream_)
               : runtime::Free(ptr);
  }
  void freeze_for_capture_(const std::shared_ptr<Block>& block) {
    if (pinned_mode_ && pin_owner_ == std::this_thread::get_id() &&
        !capture_frozen_block_set_.count(block.get())) {
      capture_frozen_blocks_.push_back(block);
      try {
        capture_frozen_block_set_.insert(block.get());
      } catch (...) {
        capture_frozen_blocks_.pop_back();
        throw;
      }
      ++block->pin_count;
    }
  }
  void release_frozen_blocks_(
      const std::vector<std::shared_ptr<Block>>& blocks) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& block : blocks)
        if (block->pin_count > 0) --block->pin_count;
    } catch (...) {
    }
  }
  std::byte* allocate_impl_(std::size_t size, bool exact) {
    if (size == 0) {
      if (exact)
        throw std::invalid_argument("exact allocator request must be nonzero");
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    size = align_up(size, 256);
    const uint64_t generation =
        pinned_mode_ && pin_owner_ == std::this_thread::get_id()
            ? capture_generation_
            : 0;
    std::shared_ptr<Block> block;
    for (std::size_t class_index = 0;
         !exact && class_index < size_classes_.size(); ++class_index) {
      auto& cls = size_classes_[class_index];
      if (size <= cls.block_size) {
        const auto free_block = std::find_if(
            cls.free_blocks.begin(), cls.free_blocks.end(),
            [](const auto& candidate) { return !candidate->in_use; });
        if (free_block != cls.free_blocks.end()) {
          block = *free_block;
          freeze_for_capture_(block);
          block->allocation_capture_generation = generation;
          cls.free_blocks.erase(free_block);
          block->in_use = true;
          block->use_count = 1;
          return reinterpret_cast<std::byte*>(block->ptr);
        }
        block = std::make_shared<Block>();
        block->size = cls.block_size;
        block->kind = Block::Kind::SizeClass;
        block->class_index = class_index;
        block->in_use = true;
        block->use_count = 1;
        try {
          check(allocate_block_(&block->ptr, block->size),
                "allocate size-class block");
          all_blocks_.emplace(block->ptr, block);
          freeze_for_capture_(block);
          block->allocation_capture_generation = generation;
        } catch (...) {
          if (block->ptr) {
            all_blocks_.erase(block->ptr);
            (void)free_block_(block->ptr);
          }
          throw;
        }
        return reinterpret_cast<std::byte*>(block->ptr);
      }
    }
    // Pinned free blocks remain reusable after their logical owners release
    // them: pinning preserves the address, not exclusive graph ownership.
    auto it = std::find_if(
        large_blocks_.begin(), large_blocks_.end(),
        [size, exact](const std::shared_ptr<Block>& candidate) {
          return !candidate->in_use &&
                 candidate->kind ==
                     (exact ? Block::Kind::Exact : Block::Kind::Large) &&
                 (exact ? candidate->size == size : candidate->size >= size);
        });
    if (it != large_blocks_.end()) {
      block = *it;
      freeze_for_capture_(block);
      block->allocation_capture_generation = generation;
      block->in_use = true;
      block->use_count = 1;
      return reinterpret_cast<std::byte*>(block->ptr);
    }
    block = std::make_shared<Block>();
    block->size = size;
    block->kind = exact ? Block::Kind::Exact : Block::Kind::Large;
    block->in_use = true;
    block->use_count = 1;
    try {
      check(allocate_block_(&block->ptr, block->size),
            "allocate large/exact block");
      all_blocks_.emplace(block->ptr, block);
      large_blocks_.push_back(block);
      freeze_for_capture_(block);
      block->allocation_capture_generation = generation;
    } catch (...) {
      if (block->ptr) {
        large_blocks_.erase(
            std::remove(large_blocks_.begin(), large_blocks_.end(), block),
            large_blocks_.end());
        all_blocks_.erase(block->ptr);
        (void)free_block_(block->ptr);
      }
      throw;
    }
    return reinterpret_cast<std::byte*>(block->ptr);
  }
  void deallocate(std::byte* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = all_blocks_.find(reinterpret_cast<void*>(ptr));
    if (it == all_blocks_.end())
      throw std::runtime_error("Pointer not allocated by this allocator");
    auto block = it->second;
    if (!block->in_use || block->use_count == 0)
      throw std::runtime_error("Double free detected in ReusableMemoryPool");
    if (block->use_count > 1) {
      --block->use_count;
      return;
    }
    if (block->kind == Block::Kind::SizeClass)
      size_classes_[block->class_index].free_blocks.push_back(block);
    block->use_count = 0;
    block->in_use = false;
  }
  std::size_t retain_allocation(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = all_blocks_.find(ptr);
    if (it == all_blocks_.end())
      throw std::runtime_error("Pointer not allocated by this allocator");
    auto block = it->second;
    if (block->kind == Block::Kind::SizeClass) {
      auto& free_blocks = size_classes_[block->class_index].free_blocks;
      free_blocks.erase(
          std::remove(free_blocks.begin(), free_blocks.end(), block),
          free_blocks.end());
    }
    block->in_use = true;
    ++block->use_count;
    return block->size;
  }
  ~State() noexcept {
    // The final shared owner, not the user-facing pool object, closes the
    // queue. Never free backing storage if completion cannot be established.
    try {
      DeviceGuard guard{device_.type(), device_.index(), true};
      if (stream_) {
        const auto status = runtime::StreamSynchronize(stream_);
        if (status != runtime::kSuccess) {
          warn(status, "drain before release; resources retained");
          return;
        }
      }
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& entry : all_blocks_)
        if (entry.second->ptr)
          warn(free_block_(entry.second->ptr), "free block");
      if (stream_) {
        const auto status = runtime::StreamSynchronize(stream_);
        if (status != runtime::kSuccess) {
          warn(status, "complete frees; stream retained");
          return;
        }
        warn(runtime::StreamDestroy(stream_), "destroy stream");
        stream_ = nullptr;
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr,
                   "ReusableMemoryPool cleanup resources retained: %s\n",
                   error.what());
    } catch (...) {
      std::fprintf(
          stderr,
          "ReusableMemoryPool cleanup resources retained: unknown error\n");
    }
  }
};

struct ReusableMemoryPool::PinState {
  std::shared_ptr<State> owner;
  std::vector<std::shared_ptr<State::Block>> blocks;
  PinState(std::shared_ptr<State> state,
           std::vector<std::shared_ptr<State::Block>> retained) noexcept
      : owner(std::move(state)), blocks(std::move(retained)) {}
  ~PinState() noexcept { owner->release_frozen_blocks_(blocks); }
};
ReusableMemoryPool::PinLease::PinLease(std::shared_ptr<PinState> state) noexcept
    : state_(std::move(state)) {}
ReusableMemoryPool::PinLease::~PinLease() noexcept = default;
ReusableMemoryPool::ReusableMemoryPool(Device device)
    : state_(std::make_shared<State>(device)) {}
ReusableMemoryPool::~ReusableMemoryPool() noexcept = default;
Device ReusableMemoryPool::device() const { return state_->device_; }
runtime::Stream ReusableMemoryPool::stream() const { return state_->stream_; }
std::byte* ReusableMemoryPool::allocate(std::size_t size) {
  DeviceGuard guard{state_->device_.type(), state_->device_.index(), true};
  return state_->allocate_impl_(size, false);
}
std::byte* ReusableMemoryPool::allocate_exact(std::size_t size) {
  DeviceGuard guard{state_->device_.type(), state_->device_.index(), true};
  return state_->allocate_impl_(size, true);
}
void ReusableMemoryPool::deallocate(std::byte* ptr) { state_->deallocate(ptr); }
std::size_t ReusableMemoryPool::retain_allocation(void* ptr) {
  return state_->retain_allocation(ptr);
}
void ReusableMemoryPool::begin_pin_mode() {
  std::lock_guard<std::mutex> lock(state_->mutex_);
  if (state_->pinned_mode_)
    throw std::runtime_error("allocator graph capture is already active");
  if (!state_->capture_frozen_blocks_.empty() ||
      !state_->capture_frozen_block_set_.empty())
    throw std::runtime_error("allocator pin transaction contains stale blocks");
  if (state_->capture_generation_ == std::numeric_limits<uint64_t>::max())
    throw std::overflow_error("allocator capture generation exhausted");
  ++state_->capture_generation_;
  state_->pinned_mode_ = true;
  state_->pin_owner_ = std::this_thread::get_id();
}
std::shared_ptr<ReusableMemoryPool::PinLease>
ReusableMemoryPool::commit_pin_mode() {
  std::vector<std::shared_ptr<State::Block>> blocks;
  {
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (!state_->pinned_mode_) return nullptr;
    if (state_->pin_owner_ != std::this_thread::get_id())
      throw std::runtime_error(
          "cannot commit allocator pin mode: another thread owns the graph "
          "capture");
    state_->pinned_mode_ = false;
    state_->pin_owner_ = {};
    blocks = std::move(state_->capture_frozen_blocks_);
    state_->capture_frozen_block_set_.clear();
  }
  try {
    auto retained = std::make_shared<PinState>(state_, std::move(blocks));
    return std::shared_ptr<PinLease>(new PinLease(std::move(retained)));
  } catch (...) {
    state_->release_frozen_blocks_(blocks);
    throw;
  }
}
void ReusableMemoryPool::cancel_pin_mode() {
  std::lock_guard<std::mutex> lock(state_->mutex_);
  if (state_->pinned_mode_ && state_->pin_owner_ != std::this_thread::get_id())
    throw std::runtime_error(
        "cannot cancel allocator pin mode: another thread owns the graph "
        "capture");
  for (const auto& block : state_->capture_frozen_blocks_)
    if (block->pin_count > 0) --block->pin_count;
  state_->capture_frozen_blocks_.clear();
  state_->capture_frozen_block_set_.clear();
  state_->pinned_mode_ = false;
  state_->pin_owner_ = {};
}
bool ReusableMemoryPool::retain_for_capture(void* ptr) {
  std::lock_guard<std::mutex> lock(state_->mutex_);
  if (!state_->pinned_mode_ || state_->pin_owner_ != std::this_thread::get_id())
    throw std::runtime_error(
        "allocator graph capture is not active on this thread");
  auto it = state_->all_blocks_.find(ptr);
  if (it == state_->all_blocks_.end())
    throw std::runtime_error("capture requires pool-owned memory");
  state_->freeze_for_capture_(it->second);
  return it->second->allocation_capture_generation !=
         state_->capture_generation_;
}
std::size_t ReusableMemoryPool::pinned_bytes() {
  std::lock_guard<std::mutex> lock(state_->mutex_);
  std::size_t bytes = 0;
  for (const auto& entry : state_->all_blocks_)
    if (entry.second->pin_count > 0) bytes += entry.second->size;
  return bytes;
}
void ReusableMemoryPool::trim() {
  DeviceGuard guard{state_->device_.type(), state_->device_.index(), true};
  std::lock_guard<std::mutex> lock(state_->mutex_);
  for (auto& cls : state_->size_classes_) {
    for (auto it = cls.free_blocks.begin(); it != cls.free_blocks.end();) {
      if ((*it)->pin_count == 0) {
        check(state_->free_block_((*it)->ptr), "trim size-class block");
        state_->all_blocks_.erase((*it)->ptr);
        it = cls.free_blocks.erase(it);
      } else
        ++it;
    }
  }
  for (auto it = state_->large_blocks_.begin();
       it != state_->large_blocks_.end();) {
    if ((*it)->pin_count == 0 && !(*it)->in_use) {
      check(state_->free_block_((*it)->ptr), "trim large/exact block");
      state_->all_blocks_.erase((*it)->ptr);
      it = state_->large_blocks_.erase(it);
    } else
      ++it;
  }
}

}  // namespace infini::rt
