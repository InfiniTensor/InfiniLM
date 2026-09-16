#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <infinirt.h>

static std::atomic<unsigned long long> launches{0};
extern "C" unsigned long long graph_launch_count() { return launches.load(); }
extern "C" infiniStatus_t infinirtGraphLuanch(infinirtGraphExec_t graph, infinirtStream_t stream) {
    using Fn = infiniStatus_t (*)(infinirtGraphExec_t, infinirtStream_t);
    // Python loads extensions RTLD_LOCAL, so RTLD_NEXT may not see InfiniRT.
    static auto real = reinterpret_cast<Fn>(dlsym(
        dlopen("libinfinirt.so", RTLD_NOW | RTLD_LOCAL), "infinirtGraphLuanch"));
    if (!real) {
        std::abort();
    }
    auto status = real(graph, stream);
    if (status == INFINI_STATUS_SUCCESS) {
        ++launches;
    }
    return status;
}
