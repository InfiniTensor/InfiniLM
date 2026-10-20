/**
 * InfiniCCL Example: AllGather (MPI + CCL Hybrid)
 *
 * This example performs a collective gather across global ranks. MPI
 * bootstraps the processes and the CCL unique ID; the collective itself uses
 * the CCL backend.
 */

#include <unistd.h>

#include <algorithm>
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

bool RunAllGatherExample(int argc, char **argv, int warmup_iter,
                         int profile_iter, size_t num_elements) {
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

  // Setup the MPI-backed communicator used to bootstrap the CCL unique ID.
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  infinicclUniqueId id;
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  CHECK_INFINI(infinicclBroadcast(&id, &id, sizeof(id), infinicclChar, 0, comm,
                                  nullptr));

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << ccl::Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  // Prepare Data
  std::vector<float> h_send(num_elements, static_cast<float>(rank + 1));
  std::vector<float> h_recv(static_cast<size_t>(size) * num_elements, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  const size_t send_bytes = num_elements * sizeof(*d_send);
  const size_t recv_bytes = send_bytes * static_cast<size_t>(size);

  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, send_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == 0) {
    std::cout << "\n=== Performing AllGather on GPU Memory ===" << std::endl;
    std::cout << "Data size: " << num_elements << " floats ("
              << send_bytes / 1024 / 1024 << " MB per rank)" << std::endl;
    std::cout << "Operation: AllGather" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  auto all_gather_call = [&]() {
    return infinicclAllGather(d_send, d_recv, num_elements, infinicclFloat32,
                              comm, nullptr);
  };

  // Warm-up and D2H transfer the answer.
  CHECK_INFINI(all_gather_call());
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(all_gather_call());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; ++i) {
    CHECK_INFINI(all_gather_call());
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  const double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation
  bool correct = true;
  for (int source_rank = 0; source_rank < size; ++source_rank) {
    const float expected = static_cast<float>(source_rank + 1);
    const size_t offset = static_cast<size_t>(source_rank) * num_elements;

    correct = Validator::ValidateResult(h_recv.data() + offset, num_elements,
                                        expected, rank) &&
              correct;
  }

  if (rank == 0) {
    const char *kGreen = "\033[32m";
    const char *kRed = "\033[31m";
    const char *kReset = "\033[0m";

    std::cout << "\n=== AllGather Results ===" << std::endl;
    std::cout << "Correct: "
              << (correct ? (kGreen + std::string("YES") + kReset)
                          : (kRed + std::string("NO") + kReset))
              << std::endl;

    std::cout << "Sample blocks: ";
    for (int source_rank = 0; source_rank < std::min(size, 4); ++source_rank) {
      const size_t offset = static_cast<size_t>(source_rank) * num_elements;
      std::cout << "[r" << source_rank << ": " << h_recv[offset] << "] ";
    }
    std::cout << std::endl;
  }

  // Metrics Reporting (Only from rank 0 for cleaner output)
  if (rank == 0) {
    Metrics metrics{elapsed, recv_bytes, size};
    metrics.Print();
  }

  // Cleanup
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));

  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == 0) {
    std::cout << "InfiniCCL finalized." << std::endl;
  }

  return correct;
}

int main(int argc, char **argv) {
  const int warmup_iters = 2;
  const int profile_iters = 20;
  const size_t num_elements = 1 << 20;

  return RunAllGatherExample(argc, argv, warmup_iters, profile_iters,
                             num_elements)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
