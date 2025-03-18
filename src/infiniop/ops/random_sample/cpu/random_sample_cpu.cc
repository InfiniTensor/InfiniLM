#include "random_sample_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../../../tensor.h"
#include <algorithm>

namespace op::random_sample::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto dt_i = result_desc->dtype();
    auto dt_p = probs_desc->dtype();

    CHECK_DTYPE(dt_i,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
    CHECK_DTYPE(dt_p, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_API_OR(result_desc->ndim(), 0,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(probs_desc->ndim(), 1,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(probs_desc->stride(0), 1,
                 return INFINI_STATUS_BAD_TENSOR_STRIDES);

    *desc_ptr = new Descriptor(
        dt_i,
        dt_p,
        probs_desc->dim(0),
        0,
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

template <typename DT>
struct ComputeType {
    using type = DT;
};

template <>
struct ComputeType<fp16_t> {
    using type = float;
};

template <class Tidx, class Tval>
struct Scheme {
    using Tcompute = typename ComputeType<Tval>::type;

    static Tcompute get(void const *ptr, size_t i) {
        return utils::cast<Tcompute, Tval>(reinterpret_cast<Tval const *>(ptr)[i]);
    }

    static void argmax(
        void *result, void const *probs, size_t n) {

        auto idx = reinterpret_cast<Tidx *>(result);
        *idx = 0;

        auto max_val = get(probs, 0);
        for (size_t i = 0; i < n; i++) {
            if (auto val = get(probs, i); val > max_val) {
                max_val = val;
                *idx = static_cast<Tidx>(i);
            }
        }
    }

    static void random(
        void *result, void const *probs, size_t n,
        float random_val, float topp, int topk, float temperature) {

        struct KVPair {
            Tidx idx;
            Tcompute val;

            bool operator<(const KVPair &other) const {
                return val > other.val;
            }
        };

        auto idx = reinterpret_cast<Tidx *>(result);
        // build & sort
        std::vector<KVPair> pairs(n);
        for (size_t i = 0; i < n; i++) {
            pairs[i] = {static_cast<Tidx>(i), get(probs, i)};
        }
        std::sort(pairs.begin(), pairs.end());
        // softmax & sum
        auto const max_val = pairs[0].val;
        pairs[0].val = 1;
        for (size_t i = 1; i < n; i++) {
            pairs[i].val = pairs[i - 1].val + std::exp((pairs[i].val - max_val) / temperature);
        }
        // topk & topp & limit
        auto const pk = pairs[std::min(static_cast<size_t>(topk), n) - 1].val,
                   pp = pairs[n - 1].val * topp,
                   plimit = random_val * std::min(pk, pp);
        // sample
        for (size_t i = 0; i < n; i++) {
            if (plimit <= pairs[i].val) {
                *idx = pairs[i].idx;
                break;
            }
        }
    }
};

template <class Tidx, class Tval>
void switch_f(
    size_t n,
    void *result, const void *probs,
    float random_val, float topp, int topk, float temperature) {
    if (random_val == 0 || topp == 0 || topk == 1 || temperature == 0) {
        Scheme<Tidx, Tval>::argmax(result, probs, n);
    } else {
        Scheme<Tidx, Tval>::random(result, probs, n, random_val, topp, topk, temperature);
    }
}

template <class Tidx>
void switch_val(
    infiniDtype_t dt_p, size_t n,
    void *result, void const *probs,
    float random_val, float topp, int topk, float temperature) {
    switch (dt_p) {
    case INFINI_DTYPE_F16:
        switch_f<Tidx, fp16_t>(n, result, probs, random_val, topp, topk, temperature);
        break;
    case INFINI_DTYPE_F32:
        switch_f<Tidx, float>(n, result, probs, random_val, topp, topk, temperature);
        break;
    case INFINI_DTYPE_F64:
        switch_f<Tidx, double>(n, result, probs, random_val, topp, topk, temperature);
        break;
    default:
        // unreachable
        std::abort();
    }
}

void switch_idx(
    infiniDtype_t dt_i, infiniDtype_t dt_p, size_t n,
    void *result, void const *probs,
    float random_val, float topp, int topk, float temperature) {

#define CASE(DT_VAL, DT_TYP)                                                             \
    case DT_VAL:                                                                         \
        switch_val<DT_TYP>(dt_p, n, result, probs, random_val, topp, topk, temperature); \
        break

    switch (dt_i) {
        CASE(INFINI_DTYPE_I8, int8_t);
        CASE(INFINI_DTYPE_I16, int16_t);
        CASE(INFINI_DTYPE_I32, int32_t);
        CASE(INFINI_DTYPE_I64, int64_t);
        CASE(INFINI_DTYPE_U8, uint8_t);
        CASE(INFINI_DTYPE_U16, uint16_t);
        CASE(INFINI_DTYPE_U32, uint32_t);
        CASE(INFINI_DTYPE_U64, uint64_t);
    default:
        // unreachable
        std::abort();
    }

#undef CASE
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {

    switch_idx(_dt_i, _dt_p, _n, result, probs, random_val, topp, topk, temperature);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::random_sample::cpu
