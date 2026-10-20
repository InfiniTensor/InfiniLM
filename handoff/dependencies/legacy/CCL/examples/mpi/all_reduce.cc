/**
 * InfiniCCL Example: AllReduce
 * * This example demonstrates the planned API for performing a
 * collective sum-reduction across multiple GPUs and nodes.
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

void RunAllReduceExample(int argc, char **argv, int warmup_iter,
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
  std::vector<float> h_recv(kNumElements, 0.0f);

  // Initialize: each rank provides its (rank + 1) as data.
  for (size_t i = 0; i < kNumElements; i++) {
    h_send[i] = static_cast<float>(rank + 1);
  }

  float *d_send, *d_recv;
  size_t total_bytes = kNumElements * sizeof(*d_send);
  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, total_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == 0) {
    std::cout << "\n=== Performing AllReduce on GPU Memory ===" << std::endl;
    std::cout << "Data size: " << kNumElements << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Operation: Sum" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up and D2H transfer the answer.
  CHECK_INFINI(infinicclAllReduce(d_send, d_recv, kNumElements,
                                  infinicclFloat32, infinicclSum, comm,
                                  nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, kNumElements * sizeof(float),
                          Rt::MemcpyDeviceToHost));

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclAllReduce(d_send, d_recv, kNumElements,
                                    infinicclFloat32, infinicclSum, comm,
                                    nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; i++) {
    CHECK_INFINI(infinicclAllReduce(d_send, d_recv, kNumElements,
                                    infinicclFloat32, infinicclSum, comm,
                                    nullptr));
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation
  float expected = 0.0f;
  for (int r = 0; r < size; r++) {
    expected += static_cast<float>(r + 1);
  }

  Validator::ValidateResult(h_recv.data(), kNumElements, expected, rank, true,
                            "AllReduce");

  // Metrics Reporting (Only from rank 0 for cleaner output)
  if (rank == 0) {
    Metrics metrics{elapsed, total_bytes, size};
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
}

int main(int argc, char **argv) {
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  RunAllReduceExample(argc, argv, warmup_iters, profile_iters, num_elements);

  return EXIT_SUCCESS;
}
