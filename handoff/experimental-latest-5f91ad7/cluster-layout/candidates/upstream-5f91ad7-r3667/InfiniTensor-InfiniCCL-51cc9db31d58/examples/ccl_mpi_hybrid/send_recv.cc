/**
 * InfiniCCL Example: Send/Recv (Ompi + CCL Hybrid)
 *
 * This example performs point-to-point `infinicclSend` and `infinicclRecv`
 * operations between global ranks. OpenMPI bootstraps the processes and the
 * CCL unique ID; the data transfer itself uses the CCL backend.
 */

#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "backend_manifest.h"
#include "device.h"
#include "infiniccl.h"
#include "runtime.h"
#include "traits.h"
#include "utils.h"

namespace ccl = infini::ccl;

void RunSendRecvExample(int argc, char **argv, int warmup_iter,
                        int profile_iter, size_t num_elements, int sender,
                        int receiver, int required_ranks, float send_value) {
  constexpr ccl::Device::Type kDevType =
      ccl::ListGetBest<ccl::DevicePriority>(ccl::EnabledDevices{});
  using Rt = ccl::Runtime<kDevType>;

  // Initialize InfiniCCL and obtain the global rank information.
  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = 0;
  int size = 0;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  // Map local rank to GPU device.
  const char *local_rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = 0;
  if (local_rank_str != nullptr) {
    local_rank = std::atoi(local_rank_str);
  }

  CHECK_RT(Rt, Rt::SetDevice(local_rank));

  if (size < required_ranks) {
    if (rank == sender) {
      std::cerr << "Send/Recv example requires at least " << required_ranks
                << " ranks." << std::endl;
    }

    CHECK_INFINI(infinicclFinalize());
    return;
  }

  // Setup the MPI-backed communicator used to bootstrap the CCL unique ID.
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  infinicclUniqueId id;
  if (rank == sender) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  CHECK_INFINI(infinicclBroadcast(&id, &id, sizeof(id), infinicclChar, sender,
                                  comm, nullptr));

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << ccl::Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  // Prepare Data
  std::vector<float> h_send(num_elements, send_value);
  std::vector<float> h_recv(num_elements, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  size_t total_bytes = num_elements * sizeof(*d_send);

  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, total_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == sender) {
    std::cout << "\n=== Performing Send/Recv on GPU Memory ===" << std::endl;
    std::cout << "Sender rank: " << sender << std::endl;
    std::cout << "Receiver rank: " << receiver << std::endl;
    std::cout << "Data size: " << num_elements << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  auto send_recv_call = [&]() {
    if (rank == sender) {
      return infinicclSend(d_send, num_elements, infinicclFloat32, receiver,
                           comm, nullptr);
    }

    if (rank == receiver) {
      return infinicclRecv(d_recv, num_elements, infinicclFloat32, sender, comm,
                           nullptr);
    }

    return infinicclSuccess;
  };

  // Warm-up and validate the first transfer.
  CHECK_INFINI(send_recv_call());
  if (rank == receiver) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));
  }

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(send_recv_call());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; i++) {
    CHECK_INFINI(send_recv_call());
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  if (rank == receiver) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));
  }

  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation
  if (rank == receiver) {
    bool correct = Validator::ValidateResult(
        h_recv.data(), num_elements, send_value, rank, false, "SendRecv");

    const char *kGreen = "\033[32m";
    const char *kRed = "\033[31m";
    const char *kReset = "\033[0m";

    std::cout << "\n=== Send/Recv Results ===" << std::endl;
    std::cout << "Correct: "
              << (correct ? (kGreen + std::string("YES") + kReset)
                          : (kRed + std::string("NO") + kReset))
              << std::endl;
    std::cout << "Expect:  " << send_value << std::endl;
    std::cout << "Actual:  " << h_recv[0] << std::endl;

    if (!correct) {
      CHECK_RT(Rt, Rt::Free(d_send));
      CHECK_RT(Rt, Rt::Free(d_recv));
      CHECK_INFINI(infinicclCommDestroy(comm));
      CHECK_INFINI(infinicclFinalize());
      std::exit(EXIT_FAILURE);
    }
  }

  // Metrics Reporting (Only from the sender for cleaner output)
  if (rank == sender) {
    Metrics metrics{elapsed, total_bytes, required_ranks};
    metrics.Print();
  }

  // Cleanup
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));

  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == sender) {
    std::cout << "InfiniCCL finalized." << std::endl;
  }
}

int main(int argc, char **argv) {
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;
  constexpr int kSender = 0;
  constexpr int kReceiver = 1;
  constexpr int kRequiredRanks = 2;
  constexpr float kSendValue = 7.0f;

  RunSendRecvExample(argc, argv, warmup_iters, profile_iters, num_elements,
                     kSender, kReceiver, kRequiredRanks, kSendValue);

  return EXIT_SUCCESS;
}
