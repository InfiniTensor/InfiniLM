#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_API_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_API_H_

#include <cncl.h>

#include <cstddef>

#include "backends/ccl/common/api.h"
#include "devices/cambricon/checks.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
struct CnclApi {
  static constexpr BackendType kBackendType = BackendType::kCncl;
  static constexpr Device::Type kDeviceType = device;

  using Comm = cnclComm_t;
  using UniqueId = cnclCliqueId;
  using Result = cnclResult_t;
  using DataType = cnclDataType_t;
  using RedOp = cnclReduceOp_t;
  using Stream = typename Runtime<device>::Stream;

 private:
  // CNCL does not support a `nullptr` queue argument, so we need to manage a
  // default queue for synchronous operations.
  struct DefaultQueue {
    Stream queue = nullptr;
    int device_id = -1;

    ~DefaultQueue() {
      if (queue != nullptr) {
        INFINI_CHECK_CNRT(cnrtQueueDestroy(queue));
      }
    }

    void SetDevice(int new_device_id) {
      if (queue != nullptr) {
        INFINI_CHECK_CNRT(cnrtQueueDestroy(queue));
      }
      INFINI_CHECK_CNRT(cnrtQueueCreate(&queue));
      device_id = new_device_id;
    }
  };

  static Stream GetDefaultQueue() {
    thread_local DefaultQueue default_queue;

    int device_id = 0;
    INFINI_CHECK_CNRT(cnrtGetDevice(&device_id));
    if (default_queue.device_id != device_id) {
      default_queue.SetDevice(device_id);
    }
    return default_queue.queue;
  }

  using PointToPointOp = Result (*)(void*, size_t, DataType, int, Comm, Stream);

  static Result PointToPoint(PointToPointOp operation, void* buffer,
                             size_t count, DataType data_type, int peer,
                             Comm comm, Stream stream) {
    // Up to CNCL 1.30.8, a `nullptr` queue argument is unsupported.
    // Synchronize the fallback queue to preserve the synchronous behavior of
    // the `nullptr` path; explicit queues stay async. Future CNCL versions may
    // remove this compatibility path.
    const bool is_default_queue = stream == nullptr;
    if (is_default_queue) {
      stream = GetDefaultQueue();
    }

    Result result = operation(buffer, count, data_type, peer, comm, stream);
    if (is_default_queue) {
      INFINI_CHECK_CNRT(cnrtQueueSync(stream));
    }
    return result;
  }

 public:
  static ReturnStatus Check(Result result) {
    if (result != CNCL_RET_SUCCESS) {
      LOG(cnclGetErrorStr(result));
      return ReturnStatus::kSystemError;
    }
    return ReturnStatus::kSuccess;
  }

  static Result GetUniqueId(UniqueId* id) { return cnclGetCliqueId(id); }

  static Result InitComms(Comm* comms, int num_comm, const int* dev_list,
                          const int* rank_list, int nrank,
                          UniqueId* clique_id) {
    return cnclInitComms(comms, num_comm, dev_list, rank_list, nrank,
                         clique_id);
  }

  static Result CommInitRank(Comm* comm, int nranks, UniqueId id, int rank) {
    using Rt = Runtime<device>;

    int device_id = 0;
    INFINI_CHECK_CNRT(Rt::GetDevice(&device_id));
    return InitComms(comm, 1, &device_id, &rank, nranks, &id);
  }

  static Result CommDestroy(Comm comm) { return cnclFreeComm(comm); }

  static Result AllReduce(const void* send_buff, void* recv_buff, size_t count,
                          DataType data_type, RedOp op, Comm comm,
                          Stream stream) {
    return cnclAllReduce(send_buff, recv_buff, count, data_type, op, comm,
                         stream);
  }

  static Result AllGather(const void* send_buff, void* recv_buff,
                          size_t send_count, DataType data_type, Comm comm,
                          Stream stream) {
    return cnclAllGather(send_buff, recv_buff, send_count, data_type, comm,
                         stream);
  }

  static Result Send(const void* send_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return PointToPoint(cnclSend, const_cast<void*>(send_buff), count,
                        data_type, peer, comm, stream);
  }

  static Result Recv(void* recv_buff, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    return PointToPoint(cnclRecv, recv_buff, count, data_type, peer, comm,
                        stream);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_API_H_
