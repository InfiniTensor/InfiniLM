#include "forward_context.hpp"

#include <memory>

namespace infinilm::engine {

namespace {

thread_local std::shared_ptr<ForwardContext> _forward_context = std::make_shared<ForwardContext>();

} // namespace

void set_forward_context(const infinilm::InfinilmModel::Input &input) {
    _forward_context->attn_metadata = input;
}

ForwardContext &get_forward_context() {
    return *_forward_context;
}

} // namespace infinilm::engine
