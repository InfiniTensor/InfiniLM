#include "tensor.h"

#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  using namespace infini::ops;

  const Tensor::Shape shape{2, 3, 4};

  const auto num_elements{
      std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>())};

  std::vector<float> elems(num_elements);

  std::iota(elems.begin(), elems.end(), 0);

  Tensor x{elems.data(), shape};

  std::cout << x.ToString() << '\n';

  return 0;
}
