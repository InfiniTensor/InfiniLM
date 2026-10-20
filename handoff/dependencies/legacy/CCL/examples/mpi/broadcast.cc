/**
 * InfiniCCL Example: Broadcast
 *
 * This example demonstrates the API for performing a collective
 * broadcast operation across multiple accelerators and nodes,
 * supporting out-of-place, in-place, and legacy in-place topologies.
 */

#include <unistd.h>

#include <functional>
#include <iostream>
#include <vector>

// Public API
#include "infiniccl.h"

// Example-Specific Utilities
#include "utils.h"

// Internal Headers
#include "backend_manifest.h"
#include "device.h"
#include "runtime.h"
#include "traits.h"

using namespace infini::ccl;

void RunBroadcastExample(int argc, char **argv, int warmup_iter,
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

  // Root of the Broadcast
  constexpr int kRoot = 0;
  constexpr float kRootMagicValue = 42.0f;

  // Prepare Host Data
  std::vector<float> h_send(kNumElements, 0.0f);
  std::vector<float> h_recv(kNumElements, 0.0f);

  // Initialize: Only root fills its buffer with meaningful payload.
  if (rank == kRoot) {
    for (size_t i = 0; i < kNumElements; i++) {
      h_send[i] = kRootMagicValue;
    }
  }

  // Allocate Device Memory
  float *d_send, *d_recv;
  size_t total_bytes = kNumElements * sizeof(float);
  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, total_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, total_bytes));

  // Copy data from host to device memory.
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // The Abstract Execution & Validation Function
  auto ProfileAndValidateScenario = [&](const std::string &scenario_name,
                                        float *d_verify_source,
                                        auto &&collective_call) {
    if (rank == kRoot) {
      std::cout << "\n=== Performing " << scenario_name << " ===" << std::endl;
      if (scenario_name.find("Scenario 1") != std::string::npos) {
        std::cout << "Data size: " << kNumElements << " floats ("
                  << total_bytes / 1024 / 1024 << " MB)" << std::endl;
        std::cout << "Root Node: " << kRoot << std::endl;
      }
    }

    // Warm-up Iteration & Initial Capture
    CHECK_INFINI(collective_call());
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_verify_source, total_bytes,
                            Rt::MemcpyDeviceToHost));

    for (int i = 1; i < warmup_iter; ++i) {
      CHECK_INFINI(collective_call());
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    // Profiling
    Timer timer;
    for (int i = 0; i < profile_iter; i++) {
      CHECK_INFINI(collective_call());
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

    // Validation
    Validator::ValidateResult(h_recv.data(), kNumElements, kRootMagicValue,
                              rank, true, scenario_name);

    // Performance Reporting
    if (rank == kRoot) {
      Metrics metrics{elapsed, total_bytes, size};
      metrics.Print();
    }
  };

  // --------------------------------------------------------------------------
  // Scenario 1: Out-of-Place Broadcast
  // --------------------------------------------------------------------------
  ProfileAndValidateScenario(
      "Scenario 1: Out-of-Place Broadcast", d_recv, [&]() {
        return infinicclBroadcast(d_send, d_recv, kNumElements,
                                  infinicclFloat32, kRoot, comm, nullptr);
      });

  // --------------------------------------------------------------------------
  // Scenario 2: In-Place Broadcast
  // --------------------------------------------------------------------------
  std::fill(h_recv.begin(), h_recv.end(), 0.0f);
  float *d_inplace_buf = (rank == kRoot) ? d_send : d_recv;

  ProfileAndValidateScenario(
      "Scenario 2: In-Place Broadcast", d_inplace_buf, [&]() {
        return infinicclBroadcast(d_inplace_buf, d_inplace_buf, kNumElements,
                                  infinicclFloat32, kRoot, comm, nullptr);
      });

  // --------------------------------------------------------------------------
  // Scenario 3: Legacy In-Place Bcast
  // --------------------------------------------------------------------------
  std::fill(h_recv.begin(), h_recv.end(), 0.0f);

  ProfileAndValidateScenario(
      "Scenario 3: Legacy In-Place Bcast", d_inplace_buf, [&]() {
        return infinicclBcast(d_inplace_buf, kNumElements, infinicclFloat32,
                              kRoot, comm, nullptr);
      });

  // --------------------------------------------------------------------------
  // Cleanup
  // --------------------------------------------------------------------------
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));

  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == kRoot) {
    std::cout << "\nInfiniCCL finalized." << std::endl;
  }
}

int main(int argc, char **argv) {
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  RunBroadcastExample(argc, argv, warmup_iters, profile_iters, num_elements);

  return EXIT_SUCCESS;
}
