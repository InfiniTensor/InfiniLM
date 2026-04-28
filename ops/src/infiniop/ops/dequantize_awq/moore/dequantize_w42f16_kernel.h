#pragma once
#include <musa_fp16.h> // 需要此头文件来支持 __half 和 __half2 类型

/**
 * @brief 将一个包含8个4-bit整数的uint32_t反量化为8个half精度浮点数。
 *
 * 这是一个通用的 CUDA C++ 实现，用于替代原有的 PTX 汇编版本，
 * 以便在不支持高级 PTX 指令（如 lop3.b32）的 GPU 上运行。
 * 输出顺序匹配 PTX 的交错打包：v0, v4, v1, v5, v2, v6, v3, v7（经 signed 调整后）。
 *
 * @param source 输入的32位无符号整数，它打包了8个4-bit的数据。
 * @return 一个 uint4 变量，其中包含8个反量化后的 half 值。
 */
__device__ __forceinline__ uint4 dequantize_s4_to_fp16x2_awq(uint32_t const &source) {
    // 步骤 1: 从一个 32-bit 源数据中解包出 8 个 4-bit 无符号整数。
    // 源数据的内存布局被假定为 [v7, v6, v5, v4, v3, v2, v1, v0]，
    // 其中每个 'v' 都是一个 4-bit 的半字节 (nibble)。
    const unsigned int v0 = (source >> 0) & 0x0F;
    const unsigned int v1 = (source >> 4) & 0x0F;
    const unsigned int v2 = (source >> 8) & 0x0F;
    const unsigned int v3 = (source >> 12) & 0x0F;
    const unsigned int v4 = (source >> 16) & 0x0F;
    const unsigned int v5 = (source >> 20) & 0x0F;
    const unsigned int v6 = (source >> 24) & 0x0F;
    const unsigned int v7 = (source >> 28) & 0x0F;

    // 步骤 2: 对于 signed 4-bit (s4)，减去 8 以映射到 [-8, 7] 范围。
    // 定义偏移量
    __half offset = __half(8);

    // 计算 signed 值
    __half hv0 = __half(v0) - offset;
    __half hv1 = __half(v1) - offset;
    __half hv2 = __half(v2) - offset;
    __half hv3 = __half(v3) - offset;
    __half hv4 = __half(v4) - offset;
    __half hv5 = __half(v5) - offset;
    __half hv6 = __half(v6) - offset;
    __half hv7 = __half(v7) - offset;

    // 步骤 3: 将 half 值按 PTX 交错顺序打包成 __half2 并存入 result 中。
    // 顺序：result_ptr[0]: low=hv0, high=hv4
    //       result_ptr[1]: low=hv1, high=hv5
    //       result_ptr[2]: low=hv2, high=hv6
    //       result_ptr[3]: low=hv3, high=hv7
    // __halves2half2 函数：low 为第一个参数，high 为第二个参数。
    uint4 result;
    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);

    result_ptr[0] = __halves2half2(hv0, hv4);
    result_ptr[1] = __halves2half2(hv1, hv5);
    result_ptr[2] = __halves2half2(hv2, hv6);
    result_ptr[3] = __halves2half2(hv3, hv7);

    return result;
}
