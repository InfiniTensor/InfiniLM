#include <cstddef>
#include <cstdint>
#include <vector>

#include <infini/rt.h>

int main() {
  namespace rt = infini::rt::runtime;

  if (rt::SetDevice(0) != rt::kSuccess) {
    return 1;
  }

  std::vector<std::uint8_t> input{1, 2, 3, 4};
  std::vector<std::uint8_t> output(input.size());
  void* device_ptr = nullptr;

  if (rt::Malloc(&device_ptr, input.size()) != rt::kSuccess) {
    return 1;
  }

  auto status = rt::Memcpy(device_ptr, input.data(), input.size(),
                           rt::kMemcpyHostToDevice);
  if (status == rt::kSuccess) {
    status = rt::Memcpy(output.data(), device_ptr, output.size(),
                        rt::kMemcpyDeviceToHost);
  }

  const auto free_status = rt::Free(device_ptr);

  if (status != rt::kSuccess || free_status != rt::kSuccess) {
    return 1;
  }

  return output == input ? 0 : 1;
}
