#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/gemm.hpp"

namespace infinicore::op {

Tensor matmul(Tensor a, Tensor b, float alpha) {
    return gemm(a, b, alpha, 0.0f);
}

void matmul_(Tensor c, Tensor a, Tensor b, float alpha) {
    Gemm::execute(c, a, b, alpha, 0.0f);
}
} // namespace infinicore::op
