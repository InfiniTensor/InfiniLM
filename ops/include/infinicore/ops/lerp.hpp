#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Lerp {
public:
    using schema_t = void (*)(Tensor, Tensor, Tensor, Tensor);
    using schema_s = void (*)(Tensor, Tensor, Tensor, float);

    static void execute(Tensor output, Tensor start, Tensor end, Tensor weight);
    static void execute(Tensor output, Tensor start, Tensor end, float weight);

    // 【核心修改】必须声明为模板函数，才能支持 dispatcher<schema_t>() 和 dispatcher<schema_s>()
    template <typename T>
    static common::OpDispatcher<T> &dispatcher();
};

Tensor lerp(Tensor start, Tensor end, Tensor weight);
Tensor lerp(Tensor start, Tensor end, float weight);

void lerp_(Tensor output, Tensor start, Tensor end, Tensor weight);
void lerp_(Tensor output, Tensor start, Tensor end, float weight);

} // namespace infinicore::op
