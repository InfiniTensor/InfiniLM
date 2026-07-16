#include "infinicore/ops/ones.hpp"

namespace infinicore::op {

common::OpDispatcher<Ones::schema> &Ones::dispatcher() {
    static common::OpDispatcher<Ones::schema> dispatcher_;
    return dispatcher_;
};

void Ones::execute(Tensor output) {
}

} // namespace infinicore::op
