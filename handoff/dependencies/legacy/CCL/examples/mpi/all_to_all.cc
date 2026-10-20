/**
 * InfiniCCL Example: AllToAll
 * * This example demonstrates the planned API for performing a
 * collective all-to-all exchange across multiple GPUs and nodes.
 */

#include <unistd.h>

#include <algorithm>
#include <cmath>
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

void RunAllToAllExample(int argc, char **argv, int warmup_iter,
                        int profile_iter, const size_t kCountPerPeer) {
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

  // Setup communicator
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Prepare Data
  // For AllToAll, `total_elements_per_rank = count_per_peer * world_size`.
  const size_t kTotalCount = kCountPerPeer * static_cast<size_t>(size);

  std::vector<float> h_send(kTotalCount);
  std::vector<float> h_recv(kTotalCount, 0.0f);

  // Layout: block per destination rank.
  // `block[dst]` is what this rank sends to dst.
  // Fill with value encoding `src`/`dst` for easy validation.
  for (int dst = 0; dst < size; ++dst) {
    float v = static_cast<float>(rank * 1000 + dst);
    size_t off = static_cast<size_t>(dst) * kCountPerPeer;
    for (size_t i = 0; i < kCountPerPeer; ++i) {
      h_send[off + i] = v;
    }
  }

  float *d_send = nullptr;
  float *d_recv = nullptr;
  size_t total_bytes = kTotalCount * sizeof(*d_send);

  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, total_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == 0) {
    std::cout << "\n=== Performing AllToAll on GPU Memory ===" << std::endl;
    std::cout << "Count per peer: " << kCountPerPeer << " floats" << std::endl;
    std::cout << "Total per rank: " << kTotalCount << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up and D2H transfer the answer.
  CHECK_INFINI(infinicclAllToAll(d_send, d_recv, kCountPerPeer,
                                 infinicclFloat32, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                          Rt::MemcpyDeviceToHost));

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, kCountPerPeer,
                                   infinicclFloat32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, kCountPerPeer,
                                   infinicclFloat32, comm, nullptr));
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                          Rt::MemcpyDeviceToHost));
  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation
  // recv `block[src]` on rank r should come from `src`'s send `block[dst=r]`.
  bool correct = true;
  int error_count = 0;

  for (int src = 0; src < size; ++src) {
    float expected = static_cast<float>(src * 1000 + rank);
    size_t off = static_cast<size_t>(src) * kCountPerPeer;

    bool block_ok = Validator::ValidateResult(h_recv.data() + off,
                                              kCountPerPeer, expected, rank);

    correct = correct && block_ok;
  }

  if (rank == 0) {
    const char *GREEN = "\033[32m";
    const char *RED = "\033[31m";
    const char *RESET = "\033[0m";

    std::cout << "\n=== AllToAll Results ===" << std::endl;
    std::cout << "Correct: "
              << (correct ? (GREEN + std::string("YES") + RESET)
                          : (RED + std::string("NO") + RESET));
    if (!correct) {
      std::cout << " (" << error_count << " errors)";
    }
    std::cout << std::endl;

    std::cout << "Sample recv blocks: ";
    for (int src = 0; src < std::min(size, 4); ++src) {
      size_t off = static_cast<size_t>(src) * kCountPerPeer;
      std::cout << "[src" << src << ": " << h_recv[off] << "] ";
    }
    std::cout << std::endl;
  }

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
  size_t count_per_peer = 1 << 18;

  RunAllToAllExample(argc, argv, warmup_iters, profile_iters, count_per_peer);

  return EXIT_SUCCESS;
}
