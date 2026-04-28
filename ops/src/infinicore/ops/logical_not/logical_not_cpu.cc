#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/logical_not.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <cstring> // memcpy
#include <stdexcept>
#include <vector>

namespace infinicore::op::logical_not_impl::cpu {

// ---------- safe load/store (avoid strict-aliasing & alignment UB) ----------
template <typename T>
static inline T load_scalar(const uint8_t *p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

template <typename T>
static inline void store_scalar(uint8_t *p, T v) {
    std::memcpy(p, &v, sizeof(T));
}

// ---------- read "truthiness": treat nonzero as true ----------
static inline bool read_truth(const uint8_t *base, ptrdiff_t elem_off, DataType dtype, size_t elem_size) {
    const uint8_t *p = base + elem_off * static_cast<ptrdiff_t>(elem_size);

    switch (dtype) {
    case DataType::BOOL:
    case DataType::U8:
        return load_scalar<uint8_t>(p) != 0;

    case DataType::I16:
        return load_scalar<int16_t>(p) != 0;
    case DataType::U16:
        return load_scalar<uint16_t>(p) != 0;

    case DataType::I32:
        return load_scalar<int32_t>(p) != 0;
    case DataType::U32:
        return load_scalar<uint32_t>(p) != 0;

    case DataType::I64:
        return load_scalar<int64_t>(p) != 0;
    case DataType::U64:
        return load_scalar<uint64_t>(p) != 0;

    case DataType::F32:
        return load_scalar<float>(p) != 0.0f;
    case DataType::F64:
        return load_scalar<double>(p) != 0.0;

    case DataType::F16: {
        fp16_t h = load_scalar<fp16_t>(p);
        float fv = utils::cast<float>(h);
        return fv != 0.0f;
    }

    default:
        throw std::runtime_error("logical_not(cpu): unsupported input dtype.");
    }
}

// ---------- write result (bool -> out_dtype as 0/1) ----------
static inline void write_bool(uint8_t *base, ptrdiff_t elem_off, DataType out_dtype, size_t out_elem_size, bool b) {
    uint8_t *p = base + elem_off * static_cast<ptrdiff_t>(out_elem_size);

    switch (out_dtype) {
    case DataType::BOOL:
    case DataType::U8:
        store_scalar<uint8_t>(p, b ? 1 : 0);
        return;

    case DataType::I16:
        store_scalar<int16_t>(p, b ? 1 : 0);
        return;
    case DataType::U16:
        store_scalar<uint16_t>(p, b ? 1 : 0);
        return;

    case DataType::I32:
        store_scalar<int32_t>(p, b ? 1 : 0);
        return;
    case DataType::U32:
        store_scalar<uint32_t>(p, b ? 1u : 0u);
        return;

    case DataType::I64:
        store_scalar<int64_t>(p, b ? 1 : 0);
        return;
    case DataType::U64:
        store_scalar<uint64_t>(p, b ? 1ull : 0ull);
        return;

    case DataType::F32:
        store_scalar<float>(p, b ? 1.0f : 0.0f);
        return;
    case DataType::F64:
        store_scalar<double>(p, b ? 1.0 : 0.0);
        return;

    case DataType::F16: {
        fp16_t h = utils::cast<fp16_t>(b ? 1.0f : 0.0f);
        store_scalar<fp16_t>(p, h);
        return;
    }

    default:
        throw std::runtime_error("logical_not(cpu): unsupported output dtype.");
    }
}

void calculate(Tensor output, Tensor input) {
    const auto ndim = output->ndim();
    if (input->ndim() != ndim) {
        throw std::runtime_error("logical_not(cpu): input/output ndim mismatch.");
    }

    const auto numel = output->numel();
    const auto shape = output->shape();

    // IMPORTANT: strides in TensorImpl are "element strides" (see tensor.cc calculate_contiguous_strides)
    const auto in_strides = input->strides();
    const auto out_strides = output->strides();

    const auto in_dtype = input->dtype();
    const auto out_dtype = output->dtype();

    const auto in_es = input->element_size();
    const auto out_es = output->element_size();

    const uint8_t *in_base = reinterpret_cast<const uint8_t *>(input->data());
    uint8_t *out_base = reinterpret_cast<uint8_t *>(output->data());

    std::vector<size_t> idx(ndim, 0);

    for (size_t linear = 0; linear < numel; ++linear) {
        // use ptrdiff_t all the way (must NOT cast strides to size_t)
        ptrdiff_t in_elem_off = 0;
        ptrdiff_t out_elem_off = 0;

        for (size_t d = 0; d < ndim; ++d) {
            in_elem_off += static_cast<ptrdiff_t>(idx[d]) * in_strides[d];
            out_elem_off += static_cast<ptrdiff_t>(idx[d]) * out_strides[d];
        }

        // logical_not: result == (input == 0) == !(truthiness)
        const bool truth = read_truth(in_base, in_elem_off, in_dtype, in_es);
        const bool result = !truth;

        write_bool(out_base, out_elem_off, out_dtype, out_es, result);

        // increment multi-d index
        for (ptrdiff_t d = static_cast<ptrdiff_t>(ndim) - 1; d >= 0; --d) {
            idx[static_cast<size_t>(d)]++;
            if (idx[static_cast<size_t>(d)] < shape[static_cast<size_t>(d)]) {
                break;
            }
            idx[static_cast<size_t>(d)] = 0;
        }
    }
}

static bool registered = []() {
    LogicalNot::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::logical_not_impl::cpu
