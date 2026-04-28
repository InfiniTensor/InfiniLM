#pragma once

__device__ uint4 dequantize_s4_to_fp16x2_awq(uint32_t const &source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
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
#else
    uint4 result;

    uint32_t *h = reinterpret_cast<uint32_t *>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const &>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is
    // thanks to the register packing format and the fact that we force our
    // integers to be unsigned, and account for this in the fp16 subtractions. In
    // addition, I exploit the fact that sub and fma have the same throughput in
    // order to convert elt_23 and elt_67 to fp16 without having to shift them to
    // the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
    // dependency if we issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                   "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                   "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                   "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM),
                   "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit
    // float2half instructions if I use the half2 ctor. In this case, I chose
    // performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    // static constexpr uint32_t NEG_72 = 0xd480d480;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[0])
                 : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[2])
                 : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[3])
                 : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    return result;
#endif
    __builtin_unreachable(); // Suppress missing return statement warning
}
