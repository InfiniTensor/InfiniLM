#include "infinicore/ops/scaled_mm_w4a16_gptq_qy.hpp"
#include "../../utils.hpp"
#include <iostream>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GptqQyblasGemm);

GptqQyblasGemm::GptqQyblasGemm(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, int64_t quant_type, int64_t bit) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, in, qweight, scales, qzeros);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, in, qweight, scales, qzeros, quant_type, bit);
}

void GptqQyblasGemm::execute(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, int64_t quant_type, int64_t bit) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GptqQyblasGemm, out, in, qweight, scales, qzeros, quant_type, bit);
}

void scaled_mm_w4a16_gptq_qy_(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, int64_t quant_type, int64_t bit) {

    GptqQyblasGemm::execute(out, in, qweight, scales, qzeros, quant_type, bit);
}

} // namespace infinicore::op
