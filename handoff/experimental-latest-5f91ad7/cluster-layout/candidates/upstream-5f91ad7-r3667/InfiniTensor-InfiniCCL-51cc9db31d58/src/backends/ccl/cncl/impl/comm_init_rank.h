#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_RANK_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_RANK_H_

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "backends/ccl/cncl/checks.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/comm_init_rank.h"
#include "communicator.h"

namespace infini::ccl {

template <Device::Type device>
class CommInitRankImpl<BackendType::kCncl, device> {
  using Api = CclApi<BackendType::kCncl, device>;
  using CommInstance = CclCommInstance<Api>;
  using CliqueId = typename Api::UniqueId;

  // CNCL initializes all communicators belonging to one process in a single
  // call, so each thread contributes one request to a local initialization
  // group.
  struct Request {
    Communicator* comm;
    CliqueId clique_id;
    int nranks;
    int rank;
    int expected_local_ranks;
    ReturnStatus status = ReturnStatus::kSuccess;
    bool done = false;
  };

  // Only one clique group may be initialized at a time in a process. Requests
  // for another group wait in `deferred_requests` until the active group ends.
  struct ThreadCoordinator {
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<Request*> active_requests;
    std::vector<Request*> deferred_requests;
    CliqueId active_clique_id{};
    int active_nranks = 0;
    int active_expected_local_ranks = 0;
    bool has_active_group = false;
  };

  static bool SameClique(const CliqueId& left, const CliqueId& right) {
    return std::memcmp(&left, &right, sizeof(CliqueId)) == 0;
  }

  static ReturnStatus ValidateRequests(const std::vector<Request*>& requests,
                                       int nranks, int expected_local_ranks) {
    // The arrays passed to `cnclInitComms` describe only this process's local
    // communicators, while `nranks` describes the global communicator.
    if (requests.size() != static_cast<size_t>(expected_local_ranks)) {
      return ReturnStatus::kInvalidArgument;
    }

    for (size_t i = 0; i < requests.size(); ++i) {
      if (!requests[i]->comm || requests[i]->comm->intra_comm() ||
          requests[i]->rank < 0 || requests[i]->rank >= nranks) {
        return ReturnStatus::kInvalidArgument;
      }
      if (i > 0 && requests[i]->rank == requests[i - 1]->rank) {
        return ReturnStatus::kInvalidArgument;
      }
    }

    std::vector<int> devices;
    devices.reserve(requests.size());
    for (const Request* request : requests) {
      devices.push_back(request->comm->device_id());
    }
    std::sort(devices.begin(), devices.end());
    for (size_t i = 1; i < devices.size(); ++i) {
      if (devices[i] == devices[i - 1]) {
        return ReturnStatus::kInvalidArgument;
      }
    }

    return ReturnStatus::kSuccess;
  }

  static bool IsSameGroup(const Request& request,
                          const ThreadCoordinator& coordinator) {
    return SameClique(request.clique_id, coordinator.active_clique_id) &&
           request.nranks == coordinator.active_nranks &&
           request.expected_local_ranks ==
               coordinator.active_expected_local_ranks;
  }

  static int ExpectedLocalRanks(const Communicator* comm, int nranks) {
    if (!comm->inter_comm()) {
      return nranks;
    }

    if (nranks == comm->size()) {
      return 1;
    }

    int local_size = comm->local_size();
    return local_size > 0 ? local_size : 1;
  }

  static void StartDeferredGroup(ThreadCoordinator& coordinator) {
    Request* first = coordinator.deferred_requests.front();
    coordinator.active_clique_id = first->clique_id;
    coordinator.active_nranks = first->nranks;
    coordinator.active_expected_local_ranks = first->expected_local_ranks;
    coordinator.has_active_group = true;

    std::vector<Request*> remaining;
    for (Request* request : coordinator.deferred_requests) {
      if (IsSameGroup(*request, coordinator)) {
        coordinator.active_requests.push_back(request);
      } else {
        remaining.push_back(request);
      }
    }
    coordinator.deferred_requests = std::move(remaining);
  }

  static void CompleteGroup(ThreadCoordinator& coordinator) {
    std::vector<Request*> requests = coordinator.active_requests;
    // Keep the arrays deterministic and align each returned CNCL handle with
    // the communicator for the corresponding global rank.
    std::sort(requests.begin(), requests.end(),
              [](const Request* left, const Request* right) {
                return left->rank < right->rank;
              });

    ReturnStatus status =
        ValidateRequests(requests, coordinator.active_nranks,
                         coordinator.active_expected_local_ranks);

    std::vector<typename Api::Comm> comms(requests.size());
    std::vector<int> devices(requests.size());
    std::vector<int> ranks(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
      ranks[i] = requests[i]->rank;
      devices[i] = requests[i]->comm->device_id();
    }

    if (status == ReturnStatus::kSuccess) {
      INFINI_CHECK_CNCL(Api::InitComms(
          comms.data(), static_cast<int>(comms.size()), devices.data(),
          ranks.data(), coordinator.active_nranks,
          &requests.front()->clique_id));
    }

    for (size_t i = 0; i < requests.size(); ++i) {
      Request* request = requests[i];
      if (status == ReturnStatus::kSuccess) {
        request->comm->set_world_info(request->rank, coordinator.active_nranks);
        auto instance = std::make_unique<CommInstance>();
        instance->handle = comms[i];
        request->comm->set_intra_comm(std::move(instance));
      }
      request->status = status;
      request->done = true;
    }

    coordinator.active_requests.clear();
    coordinator.has_active_group = false;

    if (!coordinator.deferred_requests.empty()) {
      StartDeferredGroup(coordinator);
    }
    coordinator.condition.notify_all();
  }

 public:
  static ReturnStatus Apply(Communicator* comm, int nranks,
                            infinicclUniqueId id, int rank) {
    if (!comm || comm->intra_comm() || nranks <= 0 || rank < 0 ||
        rank >= nranks) {
      return ReturnStatus::kInvalidArgument;
    }

    CliqueId clique_id{};
    std::memcpy(&clique_id, id.internal, sizeof(clique_id));

    int expected_local_ranks = ExpectedLocalRanks(comm, nranks);
    if (expected_local_ranks <= 0 || expected_local_ranks > nranks) {
      return ReturnStatus::kInvalidArgument;
    }

    // CNCL accepts each clique ID only once per process. Local ranks for the
    // same process rendezvous here so one `cnclInitComms` call creates the
    // process-local communicator set.
    static ThreadCoordinator coordinator;
    Request request{comm, clique_id, nranks, rank, expected_local_ranks};

    std::unique_lock<std::mutex> lock(coordinator.mutex);
    if (!coordinator.has_active_group) {
      coordinator.active_clique_id = clique_id;
      coordinator.active_nranks = nranks;
      coordinator.active_expected_local_ranks = expected_local_ranks;
      coordinator.has_active_group = true;
    }

    if (IsSameGroup(request, coordinator)) {
      coordinator.active_requests.push_back(&request);
    } else {
      coordinator.deferred_requests.push_back(&request);
    }

    while (!request.done) {
      if (IsSameGroup(request, coordinator) &&
          static_cast<int>(coordinator.active_requests.size()) ==
              expected_local_ranks) {
        CompleteGroup(coordinator);
      } else {
        coordinator.condition.wait(lock);
      }
    }

    return request.status;
  }
};

template <>
struct BackendEnabled<CommInitRank, BackendType::kCncl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_RANK_H_
