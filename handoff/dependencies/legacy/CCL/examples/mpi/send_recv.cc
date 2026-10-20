/**
 * InfiniCCL Example: `Send-Recv`
 *
 * This example demonstrates point-to-point `infinicclSend` and `infinicclRecv`
 * operations between rank 0 and rank 1 on accelerator memory.
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
                        int profile_iter, size_t num_elements) {
  constexpr ccl::Device::Type kDevType =
      ccl::ListGetBest<ccl::DevicePriority>(ccl::EnabledDevices{});
  using Rt = ccl::Runtime<kDevType>;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = 0;
  int size = 0;
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
            << " | GPU: " << ccl::Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  constexpr int kSender = 0;
  constexpr int kReceiver = 1;
  constexpr int kRequiredRanks = 2;
  constexpr float kSendValue = 7.0f;

  if (size < kRequiredRanks) {
    if (rank == kSender) {
      std::cerr << "Send/Recv example requires at least 2 ranks." << std::endl;
    }

    CHECK_INFINI(infinicclFinalize());
    return;
  }

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  std::vector<float> h_send(num_elements, kSendValue);
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

  if (rank == kSender) {
    std::cout << "\n=== Performing Send/Recv on GPU Memory ===" << std::endl;
    std::cout << "Sender rank: " << kSender << std::endl;
    std::cout << "Receiver rank: " << kReceiver << std::endl;
    std::cout << "Data size: " << num_elements << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  auto send_recv_call = [&]() {
    if (rank == kSender) {
      return infinicclSend(d_send, num_elements, infinicclFloat32, kReceiver,
                           comm, nullptr);
    }

    if (rank == kReceiver) {
      return infinicclRecv(d_recv, num_elements, infinicclFloat32, kSender,
                           comm, nullptr);
    }

    return infinicclSuccess;
  };

  CHECK_INFINI(send_recv_call());
  if (rank == kReceiver) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));
  }

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(send_recv_call());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;

  for (int i = 0; i < profile_iter; ++i) {
    CHECK_INFINI(send_recv_call());
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  if (rank == kReceiver) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));
  }

  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  if (rank == kReceiver) {
    bool correct = Validator::ValidateResult(
        h_recv.data(), num_elements, kSendValue, rank, false, "SendRecv");

    const char *kGreen = "\033[32m";
    const char *kRed = "\033[31m";
    const char *kReset = "\033[0m";

    std::cout << "\n=== Send/Recv Results ===" << std::endl;
    std::cout << "Correct: "
              << (correct ? (kGreen + std::string("YES") + kReset)
                          : (kRed + std::string("NO") + kReset))
              << std::endl;
    std::cout << "Expect:  " << kSendValue << std::endl;
    std::cout << "Actual:  " << h_recv[0] << std::endl;

    if (!correct) {
      CHECK_RT(Rt, Rt::Free(d_send));
      CHECK_RT(Rt, Rt::Free(d_recv));
      CHECK_INFINI(infinicclCommDestroy(comm));
      CHECK_INFINI(infinicclFinalize());
      std::exit(EXIT_FAILURE);
    }
  }

  if (rank == kSender) {
    Metrics metrics{elapsed, total_bytes, kRequiredRanks};
    metrics.Print();
  }

  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));

  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == kSender) {
    std::cout << "InfiniCCL finalized." << std::endl;
  }
}

int main(int argc, char **argv) {
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  RunSendRecvExample(argc, argv, warmup_iters, profile_iters, num_elements);

  return EXIT_SUCCESS;
}
