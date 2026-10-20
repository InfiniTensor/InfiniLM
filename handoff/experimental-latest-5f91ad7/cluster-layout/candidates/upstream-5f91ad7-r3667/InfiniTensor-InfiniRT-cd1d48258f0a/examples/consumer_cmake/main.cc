#include <cstddef>

#include <infini/rt.h>

int main() {
  namespace rt = infini::rt::runtime;

  void* ptr = nullptr;
  constexpr std::size_t size = 128;

  if (rt::SetDevice(0) != rt::kSuccess) {
    return 1;
  }
  if (rt::Malloc(&ptr, size) != rt::kSuccess) {
    return 1;
  }
  if (rt::Memset(ptr, 0, size) != rt::kSuccess) {
    rt::Free(ptr);
    return 1;
  }

  return rt::Free(ptr) == rt::kSuccess ? 0 : 1;
}
