#ifndef INFINI_OPS_HANDLE_H_
#define INFINI_OPS_HANDLE_H_

#include <cstddef>
#include <memory>

#include "cloneable.h"

namespace infini::ops {

class Handle {
 public:
  // The shared pointer owns the storage and points at its device-visible
  // address. Its deleter must defer reuse until work already enqueued on this
  // Handle's stream has finished (or use a stream-ordered allocator).
  struct WorkspaceLease {
    std::shared_ptr<void> data;
    std::size_t size_in_bytes{0};
  };

  virtual ~Handle() = default;

  virtual std::unique_ptr<Handle> Clone() const {
    return std::make_unique<Handle>(*this);
  }

  // Called by Operator::Call only after the selected/cached instance reports
  // its required size. Make and direct instance invocation do not acquire a
  // transient lease: graph owners must retain their own stable workspace.
  virtual WorkspaceLease acquire_workspace(std::size_t) const { return {}; }

  void* stream() const { return stream_; }

  void* workspace() const { return workspace_; }

  std::size_t workspace_size_in_bytes() const {
    return workspace_size_in_bytes_;
  }

  void set_stream(void* stream) { stream_ = stream; }

  void set_workspace(void* workspace) { workspace_ = workspace; }

  void set_workspace_size_in_bytes(std::size_t workspace_size_in_bytes) {
    workspace_size_in_bytes_ = workspace_size_in_bytes;
  }

 private:
  void* stream_{nullptr};

  void* workspace_{nullptr};

  std::size_t workspace_size_in_bytes_{0};
};

}  // namespace infini::ops

#endif
