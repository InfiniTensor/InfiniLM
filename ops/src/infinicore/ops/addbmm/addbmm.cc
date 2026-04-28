#include "infinicore/ops/addbmm.hpp"
#include "../../utils.hpp"
#include "infinicore/ops/addbmm.hpp"

namespace infinicore::op {

// 1. 初始化 Dispatcher
common::OpDispatcher<Addbmm::schema> &Addbmm::dispatcher() {
    static common::OpDispatcher<Addbmm::schema> dispatcher_;
    return dispatcher_;
};

void Addbmm::execute(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {

    // 切换上下文
    infinicore::context::setDevice(output->device());

    // 分发计算
    dispatcher().lookup(output->device().getType())(output, input, batch1, batch2, beta, alpha);
}

Tensor addbmm(Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    Addbmm::execute(output, input, batch1, batch2, beta, alpha);
    return output;
}

void addbmm_(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {
    Addbmm::execute(output, input, batch1, batch2, beta, alpha);
}

} // namespace infinicore::op
