#ifndef INFINI_CCL_COMMUNICATOR_H_
#define INFINI_CCL_COMMUNICATOR_H_

#include <memory>

#include "backend.h"
#include "device.h"

namespace infini::ccl {

struct BackendCommInstance {
  virtual ~BackendCommInstance() = default;
  BackendType type;
};

class Communicator {
 public:
  Communicator(Device::Type device_type, int device_id)
      : device_type_(device_type),
        device_id_(device_id),
        global_rank_(-1),
        global_size_(0) {}

  void set_world_info(int rank, int size) {
    global_rank_ = rank;
    global_size_ = size;
    return;
  }

  auto intra_comm() const { return intra_comm_.get(); }

  auto inter_comm() const { return inter_comm_.get(); }

  void set_intra_comm(std::unique_ptr<BackendCommInstance> inst) {
    intra_comm_ = std::move(inst);
    return;
  }

  void set_inter_comm(std::unique_ptr<BackendCommInstance> inst) {
    inter_comm_ = std::move(inst);
    return;
  }

  BackendType intra_comm_backend() const {
    return intra_comm_ ? intra_comm_->type : BackendType::kCount;
  }

  BackendType inter_comm_backend() const {
    return inter_comm_ ? inter_comm_->type : BackendType::kCount;
  }

  int rank() const { return global_rank_; }

  int size() const { return global_size_; }

  int device_id() const { return device_id_; }

  void set_device_id(int id) {
    device_id_ = id;
    return;
  }

  Device::Type device_type() const { return device_type_; }

  bool HasBackend(BackendType t) const {
    return (intra_comm_backend() == t) || (inter_comm_backend() == t);
  }

 private:
  // Slot 1: Intra-node (e.g., NCCL)
  std::unique_ptr<BackendCommInstance> intra_comm_;

  // Slot 2: Inter-node (e.g., OpenMPI)
  std::unique_ptr<BackendCommInstance> inter_comm_;

  int device_id_;

  int global_rank_;

  int global_size_;

  Device::Type device_type_;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_COMMUNICATOR_H_
