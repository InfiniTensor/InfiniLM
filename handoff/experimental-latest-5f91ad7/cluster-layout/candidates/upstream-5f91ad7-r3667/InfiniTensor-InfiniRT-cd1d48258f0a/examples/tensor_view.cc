#include <cstddef>
#include <vector>

#include <infini/rt.h>

int main() {
  std::vector<float> data(16);
  const infini::rt::Device device{infini::rt::Device::Type::kCpu, 0};

  const infini::rt::TensorView tensor{data.data(),
                                      std::vector<std::size_t>{4, 4},
                                      infini::rt::DataType::kFloat32, device,
                                      std::vector<std::ptrdiff_t>{4, 1}};

  if (tensor.numel() != 16) {
    return 1;
  }
  if (tensor.element_size() != sizeof(float)) {
    return 1;
  }
  if (!tensor.IsContiguous()) {
    return 1;
  }
  if (tensor.stride(0) != 4 || tensor.stride(1) != 1) {
    return 1;
  }

  return 0;
}
