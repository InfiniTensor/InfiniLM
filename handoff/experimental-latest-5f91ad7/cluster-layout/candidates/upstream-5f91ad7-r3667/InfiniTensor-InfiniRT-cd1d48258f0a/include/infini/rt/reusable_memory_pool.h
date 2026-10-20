#ifndef INFINI_RT_REUSABLE_MEMORY_POOL_H_
#define INFINI_RT_REUSABLE_MEMORY_POOL_H_

#include <infini/rt.h>

#include <cstddef>
#include <memory>

namespace infini::rt {

// Block reuse and capture retention, independent of models/requests. The pool
// owns one ordinary execution stream; consumers borrow stream(), never destroy
// it. Raw allocation pointers require a live pool or a committed pin lease.
// Reuse/deallocation is ordered on this stream by the caller. A foreign-thread
// release must first prove completion; uses on other streams must be joined.
class ReusableMemoryPool {
  struct State;
  struct PinState;

 public:
  class PinLease {
   public:
    ~PinLease() noexcept;
    PinLease(const PinLease&) = delete;
    PinLease& operator=(const PinLease&) = delete;

   private:
    friend class ReusableMemoryPool;
    explicit PinLease(std::shared_ptr<PinState> state) noexcept;
    std::shared_ptr<PinState> state_;
  };

  explicit ReusableMemoryPool(Device device);
  ~ReusableMemoryPool() noexcept;
  ReusableMemoryPool(const ReusableMemoryPool&) = delete;
  ReusableMemoryPool& operator=(const ReusableMemoryPool&) = delete;

  Device device() const;
  runtime::Stream stream() const;
  // During native CUDA stream capture, only cached blocks may be reused.
  // Preallocate before capture; a cache miss throws without invalidating it.
  std::byte* allocate(std::size_t size);
  std::byte* allocate_exact(std::size_t size);
  void deallocate(std::byte* ptr);
  // Retain an additional logical owner and return the physical block size.
  std::size_t retain_allocation(void* ptr);

  void begin_pin_mode();
  std::shared_ptr<PinLease> commit_pin_mode();
  void cancel_pin_mode();
  // True if the logical allocation predates this pin transaction.
  bool retain_for_capture(void* ptr);
  std::size_t pinned_bytes();
  void trim();

 private:
  std::shared_ptr<State> state_;
};

}  // namespace infini::rt
#endif
