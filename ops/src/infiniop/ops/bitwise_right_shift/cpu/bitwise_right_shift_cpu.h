#ifndef __BITWISE_RIGHT_SHIFT_CPU_H__
#define __BITWISE_RIGHT_SHIFT_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cstdint>
#include <type_traits>

ELEMENTWISE_DESCRIPTOR(bitwise_right_shift, cpu)

namespace op::bitwise_right_shift::cpu {
typedef struct BitwiseRightShiftOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &x, const T &shift) const {
        constexpr unsigned kBits = static_cast<unsigned>(sizeof(T) * 8);
        using WideUnsigned = std::conditional_t<(kBits <= 32), uint32_t, uint64_t>;
        using WideSigned = std::conditional_t<(kBits <= 32), int32_t, int64_t>;

        if constexpr (std::is_signed_v<T>) {
            const WideSigned xw = static_cast<WideSigned>(x);
            const WideSigned sw = static_cast<WideSigned>(shift);

            if (sw < 0 || sw >= static_cast<WideSigned>(kBits)) {
                return static_cast<T>(xw < 0 ? WideSigned(-1) : WideSigned(0));
            }

            return static_cast<T>(xw >> static_cast<unsigned>(sw));
        } else {
            const WideUnsigned xw = static_cast<WideUnsigned>(x);
            const WideUnsigned sw = static_cast<WideUnsigned>(shift);

            if (sw >= static_cast<WideUnsigned>(kBits)) {
                return static_cast<T>(0);
            }

            return static_cast<T>(xw >> static_cast<unsigned>(sw));
        }
    }
} BitwiseRightShiftOp;
} // namespace op::bitwise_right_shift::cpu

#endif // __BITWISE_RIGHT_SHIFT_CPU_H__
