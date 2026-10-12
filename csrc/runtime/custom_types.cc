#include "custom_types.h"
#include <infini/rt.h>

float _f16_to_f32(fp16_t val) {
    return infini::rt::Float16{val._v}.ToFloat();
}

fp16_t _f32_to_f16(float val) {
    return fp16_t{infini::rt::Float16::FromFloat(val).bits};
}

float _bf16_to_f32(bf16_t val) {
    return infini::rt::BFloat16{val._v}.ToFloat();
}

bf16_t _f32_to_bf16(float val) {
    return bf16_t{infini::rt::BFloat16::FromFloat(val).bits};
}
