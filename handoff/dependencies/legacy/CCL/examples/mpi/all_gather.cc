/**
 * InfiniCCL Example: AllGather
 * * This example demonstrates the planned API for performing a
 * collective data gathering across multiple GPUs and nodes.
 */

#include <unistd.h>

#include <iostream>
#include <vector>

// Public API
#include "infiniccl.h"

// Example-Specific Utilities
#include "utils.h"

// Internal Headers (Accessible via example-specific include paths, technically
// not public APIs)
#include "backend_manifest.h"
#include "device.h"
#include "runtime.h"
#include "traits.h"

using namespace infini::ccl;

bool RunAllGatherExample(int argc, char **argv, int warmup_iter,
                         int profile_iter, const size_t kNumElements) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank, size;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  // Map local rank to GPU device.
  // Note: this is just for info printing. In practice, this part is not needed.
  const char *local_rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = 0;
  if (local_rank_str != nullptr) {
    local_rank = std::atoi(local_rank_str);
  }

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  // Setup Communicator
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Prepare Data
  std::vector<float> h_send(kNumElements);
  std::vector<float> h_recv(kNumElements * size, 0.0f);

  // Initialize: each rank provides its (rank + 1) as data.
  for (size_t i = 0; i < kNumElements; i++) {
    h_send[i] = static_cast<float>(rank + 1);
  }

  float *d_send, *d_recv;
  size_t send_bytes = kNumElements * sizeof(*d_send);
  size_t recv_bytes = send_bytes * size;
  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, send_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == 0) {
    std::cout << "\n=== Performing AllGather on GPU Memory ===" << std::endl;
    std::cout << "Data size: " << kNumElements << " floats ("
              << send_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Operation: AllGather" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up and D2H transfer the answer.
  CHECK_INFINI(infinicclAllGather(d_send, d_recv, kNumElements,
                                  infinicclFloat32, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclAllGather(d_send, d_recv, kNumElements,
                                    infinicclFloat32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; i++) {
    CHECK_INFINI(infinicclAllGather(d_send, d_recv, kNumElements,
                                    infinicclFloat32, comm, nullptr));
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation
  bool correct = true;

  for (int src_rank = 0; src_rank < size; ++src_rank) {
    float expected = static_cast<float>(src_rank + 1);
    size_t offset = static_cast<size_t>(src_rank) * kNumElements;

    correct = Validator::ValidateResult(h_recv.data() + offset, kNumElements,
                                        expected, rank) &&
              correct;
  }

  if (rank == 0) {
    const char *GREEN = "\033[32m";
    const char *RED = "\033[31m";
    const char *RESET = "\033[0m";

    std::cout << "\n=== AllGather Results ===" << std::endl;
    std::cout << "Correct: "
              << (correct ? (GREEN + std::string("YES") + RESET)
                          : (RED + std::string("NO") + RESET));
    std::cout << std::endl;

    std::cout << "Sample blocks: ";
    for (int src_rank = 0; src_rank < std::min(size, 4); ++src_rank) {
      size_t offset = static_cast<size_t>(src_rank) * kNumElements;
      std::cout << "[r" << src_rank << ": " << h_recv[offset] << "] ";
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
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  return RunAllGatherExample(argc, argv, warmup_iters, profile_iters,
                             num_elements)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
