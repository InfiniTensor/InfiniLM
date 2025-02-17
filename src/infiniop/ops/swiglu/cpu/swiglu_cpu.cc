#include "swiglu_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <cmath>
#include <cstdlib>

infiniopStatus_t cpuCreateSwiGLUDescriptor(
    infiniopCpuHandle_t handle,
    infiniopSwiGLUCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto const out = c_desc,
               up = a_desc,
               gate = b_desc;

    auto dtype = out->dtype;

    // Check dtypes

    constexpr infiniDtype_t SUPPORTED_DTYPES[] = {
        INFINI_DTYPE_F16,
        INFINI_DTYPE_F32,
        INFINI_DTYPE_F64,
    };
    auto supported = false;
    for (auto supported_dtype : SUPPORTED_DTYPES) {
        if (dtype == supported_dtype) {
            supported = true;
            break;
        }
    }
    if (!supported || gate->dtype != dtype || up->dtype != dtype) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check shapes

    if (out->ndim != 2 || gate->ndim != 2 || up->ndim != 2) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }
    auto const n = out->shape[0],
               d = out->shape[1],
               n_g = gate->shape[0],
               d_g = gate->shape[1],
               n_u = up->shape[0],
               d_u = up->shape[1];
    if (n_g != n || n_u != n || d_g != d || d_u != d) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }

    // Create descriptor

    *desc_ptr = new SwiGLUCpuDescriptor{
        INFINI_DEVICE_CPU,
        dtype,
        n,
        d,
        out->strides[0],
        out->strides[1],
        gate->strides[0],
        gate->strides[1],
        up->strides[0],
        up->strides[1],
    };
    return INFINIOP_STATUS_SUCCESS;
}

template <class T>
T sigmoid(T x) {
    return 1 / (1 + std::exp(-x));
}

template <class T>
T swiglu(T gate, T up) {
    return gate * sigmoid(gate) * up;
}

template <class T>
void swiglu_ptr(uint8_t *out, uint8_t const *gate, uint8_t const *up) {
    auto out_ = reinterpret_cast<T *>(out);
    auto gate_ = reinterpret_cast<T const *>(gate);
    auto up_ = reinterpret_cast<T const *>(up);
    *out_ = swiglu(*gate_, *up_);
}

template <>
void swiglu_ptr<uint16_t>(uint8_t *out, uint8_t const *gate, uint8_t const *up) {
    auto out_ = reinterpret_cast<uint16_t *>(out);
    auto gate_ = reinterpret_cast<uint16_t const *>(gate);
    auto up_ = reinterpret_cast<uint16_t const *>(up);
    *out_ = f32_to_f16(swiglu(f16_to_f32(*gate_), f16_to_f32(*up_)));
}

infiniopStatus_t cpuSwiGLU(
    infiniopSwiGLUCpuDescriptor_t desc,
    void *c, void const *a, void const *b) {

    auto out = reinterpret_cast<uint8_t *>(c);
    auto up = reinterpret_cast<uint8_t const *>(a);
    auto gate = reinterpret_cast<uint8_t const *>(b);
    auto const unit = infiniSizeof(desc->dtype);

    for (size_t i = 0; i < desc->n; ++i) {
        for (size_t j = 0; j < desc->d; ++j) {
            auto out_ = out + (i * desc->s_no + j * desc->s_do) * unit;
            auto gate_ = gate + (i * desc->s_ng + j * desc->s_dg) * unit;
            auto up_ = up + (i * desc->s_nu + j * desc->s_du) * unit;

            switch (desc->dtype) {
            case INFINI_DTYPE_F16:
                swiglu_ptr<uint16_t>(out_, gate_, up_);
                break;
            case INFINI_DTYPE_F32:
                swiglu_ptr<float>(out_, gate_, up_);
                break;
            case INFINI_DTYPE_F64:
                swiglu_ptr<double>(out_, gate_, up_);
                break;
            default:
                // unreachable
                std::abort();
            }
        }
    }
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroySwiGLUDescriptor(
    infiniopSwiGLUCpuDescriptor_t desc) {
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
