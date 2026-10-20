#include <cstddef>

#include <infini/rt.h>

int main() {
  namespace rt = infini::rt::runtime;

  infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu);

  if (infini::rt::runtime_device_type() != infini::rt::Device::Type::kCpu) {
    return 1;
  }

  void* ptr = nullptr;
  constexpr std::size_t size = 256;

  if (rt::Malloc(&ptr, size) != rt::kSuccess) {
    return 1;
  }

  return rt::Free(ptr) == rt::kSuccess ? 0 : 1;
}
