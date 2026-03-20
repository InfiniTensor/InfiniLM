#include "forward_context.hpp"

namespace infinilm::engine {

namespace {

// One TLS object per OS thread; single definition in this TU (not duplicated
// across translation units unlike `static` in a header).
thread_local ForwardContext _forward_context;

} // namespace

void set_forward_context(const infinilm::InfinilmModel::Input &input) {
    _forward_context.attn_metadata = input;
}

ForwardContext &get_forward_context() {
    return _forward_context;
}

} // namespace infinilm::engine
